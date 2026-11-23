"""
Receivers that keep criteria groups in step with deleted criteria and archived associations, tags, and taxonomies.

A ``CompetencyCriterion`` is deleted by cascade when its ``ObjectTag`` is deleted, which is what
happens when Studio deletes a subsection or ``tag_object()`` removes a tag. The criterion's group,
and the course-level group and root above it, are then left with nothing under them. The receivers
here walk that lineage and delete or archive each group the same way
``delete_or_archive_competency_criterion()`` does.

A tagging-layer delete removes an ``ObjectTag`` only while it is unlocked.
``lock_object_tags_for_deletion()`` locks the associations a learner status depends on, and
``_reconcile_object_tags()`` locks those that only archived criteria still use. Deleting a locked
association archives it instead, which cascades into no criteria, so none of the delete receivers fires.
The tagging layer sends ``object_tags_archived`` for it instead, and ``_associations_archived()``
archives the criteria that still use the association and each group that archiving them leaves empty.

The delete receivers react only to a delete whose origin is an ``ObjectTag`` or a ``CourseRun``. An ``ObjectTag``
always arrives as a queryset, because the tagging layer deletes through ``ObjectTagQuerySet``; a ``CourseRun``
arrives as an instance or as a queryset. Deleting a ``CourseRun`` cascades to its course-level groups,
which leaves the shared root without a course beneath it and the ``ObjectTag`` of each deleted criterion
possibly unused. A criterion deleted for any other reason, including by
``delete_or_archive_competency_criterion()`` itself, is handled by whoever deleted it.

The receivers never roll back the deletion or the archive that triggered them: a failure while cleaning up is logged
and the groups stay in place. They also add no knowledge of CBE to ``openedx_tagging``; the dependency
points the other way, from this applet to the tagging models.
"""
import logging

from django.db import OperationalError, transaction
from django.db.models import Model, Q, QuerySet
from django.db.models.signals import post_delete, pre_delete
from django.dispatch import receiver

from openedx_catalog.models_api import CourseRun
from openedx_tagging.models import ObjectTag, Tag
from openedx_tagging.signals import object_tags_archived

from .api import _cascade_remove_from_group, _reconcile_object_tags, _write_criterion_archived
from .models import CompetencyCriteriaGroup, CompetencyCriterion

log = logging.getLogger(__name__)


def _is_delete_of(origin: object, model: type[Model]) -> bool:
    """
    Return whether a delete started from an instance of ``model`` or from a queryset of it.

    ``origin`` is the ``origin`` argument Django passes to ``post_delete`` receivers. The check
    queries nothing.
    """
    if isinstance(origin, QuerySet):
        return origin.model is model
    return isinstance(origin, model)


def _lineage_state(group: CompetencyCriteriaGroup) -> dict[int, bool]:
    """Return ``{id: archived}`` for ``group``, its parent and its grandparent, in that order, skipping missing ones."""
    state: dict[int, bool] = {}
    current: CompetencyCriteriaGroup | None = group
    for _ in range(3):
        if current is None:
            break
        state[current.id] = current.archived
        current = current.parent
    return state


def _clean_up_from(
    group_id: int, trigger: str, *, archive_object_tag_ids: set[int] | None = None, using: str | None = None
) -> None:
    """
    Walk up from ``group_id``, deleting or archiving each group left with no live children, and log the outcome.

    When ``archive_object_tag_ids`` is given, the live criteria of the group that use those associations are
    archived first, under the same lock and in the same savepoint as the walk, so a failure leaves them live.

    ``using`` is the alias of the database the triggering write went to. Every query and write here runs on it,
    because the router may send reads to a replica that has not yet seen that write. When ``using`` is
    ``None``, the database router chooses.

    Does nothing if the group is already gone.
    Any failure other than a database ``OperationalError`` is logged and swallowed.
    """
    groups = CompetencyCriteriaGroup.objects.using(using).select_related("parent__parent")
    group = groups.filter(pk=group_id).first()
    if group is None:
        return
    before: dict[int, bool] = {}
    try:
        with transaction.atomic(using=using):
            # The same lock, for the same three races, as delete_or_archive_competency_criterion(); see its comment.
            if Tag.objects.using(using).select_for_update().filter(pk=group.tag_id).first() is None:
                return
            # Re-read under the lock: a concurrent cleanup may have removed or changed the lineage.
            group = groups.filter(pk=group_id).first()
            if group is None:
                return
            before = _lineage_state(group)
            if archive_object_tag_ids:
                live_criteria = CompetencyCriterion.objects.using(using).filter(
                    group_id=group.pk, object_tag_id__in=archive_object_tag_ids, archived=False
                )
                for criterion in live_criteria:
                    _write_criterion_archived(criterion, None, using)
            _cascade_remove_from_group(group, None, using)
    except OperationalError:
        # A deadlock or a lock wait timeout has left the transaction unusable, so swallowing it would only move
        # the error to the next statement.
        raise
    except Exception:  # pylint: disable=broad-exception-caught
        # A failure here must not roll back the deletion that triggered the signal. The savepoint undoes
        # any partial work, so the groups are left as they were.
        log.warning(
            "Group cleanup failed after %s; CompetencyCriteriaGroups left in place: %s",
            trigger,
            sorted(before) or [group_id],
            exc_info=True,
        )
        return
    after = dict(CompetencyCriteriaGroup.objects.using(using).filter(pk__in=before).values_list("pk", "archived"))
    for pk, was_archived in before.items():
        if pk not in after:
            log.info("Group cleanup removed CompetencyCriteriaGroup %s after %s", pk, trigger)
        elif after[pk] and not was_archived:
            log.info("Group cleanup archived CompetencyCriteriaGroup %s after %s", pk, trigger)


@receiver(object_tags_archived, sender=ObjectTag)
def _associations_archived(object_tag_ids: list[int], using: str | None = None, **kwargs) -> None:
    """
    Archive the live criteria that use newly archived associations, and each group left with no live children.

    The tagging layer sends this signal, inside the transaction that archived the associations, when it removes a
    deletion-locked ``ObjectTag``. The criteria on that association survive the removal, so their rule is archived
    here. Re-applying the tag revives the association and leaves the rule archived.

    The live criteria are read before any lock is taken, restricted to associations that are archived, and the
    receiver returns when there are none. That makes the signals sent from ``_reconcile_object_tags()`` cost one
    query, because the code that calls it has already archived its criteria. Each group is then cleaned up under
    its own savepoint and ``Tag`` lock, in ``(tag_id, group_id)`` order. The same ids may arrive twice; the second
    pass finds no live criteria and changes nothing.

    The read that finds the live criteria is not inside a savepoint. A database error in it propagates and rolls
    back the archive that sent the signal, which only a test can produce.

    ``using`` is the database alias the archive was written on. All of the receiver's queries and writes run on it.
    """
    object_tag_ids_by_group: dict[tuple[int, int], set[int]] = {}
    for tag_id, group_id, object_tag_id in (
        CompetencyCriterion.objects.using(using)
        .filter(object_tag_id__in=object_tag_ids, object_tag__archived=True, archived=False)
        .values_list("group__tag_id", "group_id", "object_tag_id")
    ):
        object_tag_ids_by_group.setdefault((tag_id, group_id), set()).add(object_tag_id)
    archived_ids = {object_tag_id for ids in object_tag_ids_by_group.values() for object_tag_id in ids}
    for (_tag_id, group_id), group_object_tag_ids in sorted(object_tag_ids_by_group.items()):
        _clean_up_from(
            group_id,
            f"ObjectTags {sorted(group_object_tag_ids)} archived",
            archive_object_tag_ids=archived_ids,
            using=using,
        )


@receiver(post_delete, sender=CompetencyCriterion)
def _criterion_deleted(instance: CompetencyCriterion, origin=None, using: str | None = None, **kwargs) -> None:
    """
    Clean up the groups above a criterion that was deleted together with its ``ObjectTag``.

    ``using`` is the database alias Django deleted the criterion on; the cleanup runs on it.

    Fires only for an unlocked ``ObjectTag``: deleting a locked one archives it instead, so its criteria are
    never deleted.
    """
    if not _is_delete_of(origin, ObjectTag):
        return
    _clean_up_from(
        instance.group_id,
        f"CompetencyCriterion {instance.pk} deleted with ObjectTag {instance.object_tag_id}",
        using=using,
    )


@receiver(pre_delete, sender=CompetencyCriteriaGroup)
def _course_group_about_to_be_deleted(
    instance: CompetencyCriteriaGroup, origin=None, using: str | None = None, **kwargs
) -> None:
    """
    Remember which ``ObjectTag`` ids a course-level group's criteria use, before they are deleted.

    Django deletes the criteria before the groups, so ``post_delete`` can no longer read them.
    Django passes the same instance object to both signals, so the ids are kept on it. ``using`` is the database
    alias Django is deleting on; the ids are read from it.
    """
    if not _is_delete_of(origin, CourseRun) or instance.course_id is None:
        return
    instance._cleanup_object_tag_ids = set(  # type: ignore[attr-defined]  # pylint: disable=protected-access
        CompetencyCriterion.objects.using(using)
        .filter(Q(group_id=instance.pk) | Q(group__parent_id=instance.pk))
        .values_list("object_tag_id", flat=True)
    )


@receiver(post_delete, sender=CompetencyCriteriaGroup)
def _course_group_deleted(
    instance: CompetencyCriteriaGroup, origin=None, using: str | None = None, **kwargs
) -> None:
    """
    Clean up the root above a course-level group deleted with its ``CourseRun``, and the tags its criteria used.

    The tag step hands the stashed ids to ``_reconcile_object_tags()``, which keeps, archives, or deletes
    each association; its docstring states which outcome applies when.

    The two steps are independent and best-effort. Reconciling runs even when the root is already gone, which
    happens when two runs under one root are deleted together: each course-level group stashed only its own ids.

    ``using`` is the database alias Django deleted the group on; both steps run on it.
    """
    if not _is_delete_of(origin, CourseRun) or instance.course_id is None:
        return
    trigger = f"CompetencyCriteriaGroup {instance.pk} deleted with CourseRun {instance.course_id}"
    if instance.parent_id is not None:
        _clean_up_from(instance.parent_id, trigger, using=using)
    stashed_ids: set[int] = getattr(instance, "_cleanup_object_tag_ids", set())
    try:
        with transaction.atomic(using=using):
            # The same lock, for the same three races, as delete_or_archive_competency_criterion(); see its comment.
            # The walk above may have returned before taking it, so it is taken here too. Taking it again when the
            # walk already holds it is harmless.
            Tag.objects.using(using).select_for_update().filter(pk=instance.tag_id).first()
            _reconcile_object_tags(stashed_ids, using)
    except OperationalError:
        raise
    except Exception:  # pylint: disable=broad-exception-caught
        # As in _clean_up_from(), a failure here must not roll back the course deletion that triggered the signal.
        log.warning(
            "Group cleanup failed after %s; ObjectTags left in place: %s",
            trigger,
            sorted(stashed_ids),
            exc_info=True,
        )
