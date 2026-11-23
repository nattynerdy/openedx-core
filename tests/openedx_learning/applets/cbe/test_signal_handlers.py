"""
Tests for the receivers that keep criteria groups in step with the deletion of their criteria, and with the
archiving of associations, tags, and taxonomies.
"""
import json
import logging
from collections.abc import Callable, Iterator
from io import BytesIO

import pytest
import rules
from django.apps import apps
from django.db import OperationalError
from django.db.models import ProtectedError
from django.test import override_settings
from rules.permissions import permissions as rule_permissions

from openedx_catalog.models import CourseRun
from openedx_learning.api import delete_or_archive_competency_criteria_group, delete_or_archive_competency_criterion
from openedx_learning.applets.cbe import signal_handlers
from openedx_learning.applets.cbe.api import _archive_competency_criteria_subtree
from openedx_learning.models import (
    CompetencyCriteriaGroup,
    CompetencyCriterion,
    CompetencyRuleProfile,
    CompetencyTaxonomy,
    MasteryStatus,
    StudentCompetencyCriteriaGroupStatus,
    StudentCompetencyCriterionStatus,
)
from openedx_tagging.api import delete_object_tags, delete_taxonomy, lock_object_tags_for_deletion, tag_object
from openedx_tagging.import_export import ParserFormat
from openedx_tagging.import_export.api import import_tags
from openedx_tagging.models import ObjectTag, Tag, Taxonomy
from openedx_tagging.signals import object_tags_archived, tags_archived, taxonomies_archived
from tests.openedx_tagging.utils import WritesToPrimaryRouter

pytestmark = pytest.mark.django_db

CHANGE_OBJECTTAG_OBJECTID = "oel_tagging.change_objecttag_objectid"


@pytest.fixture(autouse=True)
def _allow_object_level_tagging() -> Iterator[None]:
    """
    Let any user tag any object; restore the real rule afterwards.

    openedx_tagging denies change_objecttag_objectid to everyone and leaves the real Studio-role
    check to openedx-platform, so these tests substitute a permissive one.
    """
    original = rule_permissions[CHANGE_OBJECTTAG_OBJECTID]
    rules.set_perm(CHANGE_OBJECTTAG_OBJECTID, lambda _user, _object_id: True)
    yield
    rules.set_perm(CHANGE_OBJECTTAG_OBJECTID, original)


def _make_object_tag(competency_taxonomy: CompetencyTaxonomy, tag: Tag, object_id: str) -> ObjectTag:
    """Create an ObjectTag for `object_id`, distinct from the conftest `object_tag` fixture's own row."""
    return ObjectTag.objects.create(object_id=object_id, taxonomy=competency_taxonomy, tag=tag)


def _make_criterion(
    group: CompetencyCriteriaGroup, object_tag: ObjectTag, default_rule_profile: CompetencyRuleProfile
) -> CompetencyCriterion:
    """Create a CompetencyCriterion under `group`, on the system-default rule profile."""
    return CompetencyCriterion.objects.create(group=group, object_tag=object_tag, rule_profile=default_rule_profile)


def _tree(
    tag: Tag, course_run: CourseRun
) -> tuple[CompetencyCriteriaGroup, CompetencyCriteriaGroup, CompetencyCriteriaGroup]:
    """Return (root, course_level, leaf) for `tag` scoped to `course_run`."""
    root = CompetencyCriteriaGroup.objects.create(tag=tag)
    course_level = CompetencyCriteriaGroup.objects.create(tag=tag, parent=root, course=course_run)
    leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=course_level)
    return root, course_level, leaf


def _cleanup_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the messages the cleanup receivers logged."""
    return [r.getMessage() for r in caplog.records if r.name == signal_handlers.__name__]


def _ids_named_in(lines: list[str]) -> list[int]:
    """Return the group id each log line names, in order."""
    return [int(line.split("CompetencyCriteriaGroup ")[1].split(" ")[0]) for line in lines]


def test_tag_object_removal_removes_the_whole_empty_lineage_and_logs_each_group(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """An ObjectTag removed through tag_object() takes its criterion and every group it leaves empty."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    trigger = f"CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        # tag_object() deletes through a queryset, so the origin is a QuerySet[ObjectTag].
        tag_object(object_tag.object_id, competency_taxonomy, [])

    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf.pk]).exists()
    assert _cleanup_lines(caplog) == [
        f"Group cleanup removed CompetencyCriteriaGroup {leaf.pk} after {trigger}",
        f"Group cleanup removed CompetencyCriteriaGroup {course_level.pk} after {trigger}",
        f"Group cleanup removed CompetencyCriteriaGroup {root.pk} after {trigger}",
    ]


def test_a_sibling_leaf_with_a_criterion_stops_the_walk_and_logs_only_the_removed_leaf(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """A sibling leaf that still holds a criterion keeps the course-level group and the root in place."""
    root, course_level, leaf_a = _tree(tag, course_run)
    leaf_b = CompetencyCriteriaGroup.objects.create(tag=tag, parent=course_level)
    criterion = _make_criterion(leaf_a, object_tag, default_rule_profile)
    other_object_tag = _make_object_tag(competency_taxonomy, tag, "block-v1:Org1+Python100+Fall2026+problem+p2")
    _make_criterion(leaf_b, other_object_tag, default_rule_profile)
    trigger = f"CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert not CompetencyCriteriaGroup.objects.filter(pk=leaf_a.pk).exists()
    assert CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf_b.pk]).count() == 3
    assert _cleanup_lines(caplog) == [f"Group cleanup removed CompetencyCriteriaGroup {leaf_a.pk} after {trigger}"]


def test_queryset_origin_removes_both_leaves_the_course_group_and_the_root(
    *, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """One ObjectTag with criteria in two leaves empties both leaves, the course-level group and the root."""
    root, course_level, leaf_1 = _tree(tag, course_run)
    leaf_2 = CompetencyCriteriaGroup.objects.create(tag=tag, parent=course_level)
    _make_criterion(leaf_1, object_tag, default_rule_profile)
    _make_criterion(leaf_2, object_tag, default_rule_profile)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)  # a QuerySet[ObjectTag] origin

    group_ids = [root.pk, course_level.pk, leaf_1.pk, leaf_2.pk]
    assert not CompetencyCriteriaGroup.objects.filter(pk__in=group_ids).exists()
    lines = _cleanup_lines(caplog)
    # The order of post_delete calls within a batch is an implementation detail of Django's collector, so
    # the test compares sets.
    assert len(lines) == 4
    assert all(line.startswith("Group cleanup removed CompetencyCriteriaGroup ") for line in lines)
    assert set(_ids_named_in(lines)) == set(group_ids)


def test_two_criteria_in_one_leaf_remove_and_log_the_leaf_once(
    *, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """The second criterion's receiver finds the lineage already gone and logs nothing."""
    root, course_level, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    _make_criterion(leaf, object_tag, default_rule_profile)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf.pk]).exists()
    named = _ids_named_in(_cleanup_lines(caplog))
    assert sorted(named) == sorted([root.pk, course_level.pk, leaf.pk])
    assert all(named.count(pk) == 1 for pk in (root.pk, course_level.pk, leaf.pk))


def test_two_competencies_on_one_subsection_clean_up_independently(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """Deleting a subsection's tags removes the lineage that is now empty and keeps the one still in use."""
    tag_b = Tag.objects.create(taxonomy=competency_taxonomy, value="Revising Poetry")
    root_a, course_a, leaf_a = _tree(tag, course_run)
    root_b, course_b, leaf_b = _tree(tag_b, course_run)
    object_tag_b = _make_object_tag(competency_taxonomy, tag_b, object_tag.object_id)
    other_object_tag = _make_object_tag(competency_taxonomy, tag_b, "block-v1:Org1+Python100+Fall2026+problem+p2")
    _make_criterion(leaf_a, object_tag, default_rule_profile)
    _make_criterion(leaf_b, object_tag_b, default_rule_profile)
    _make_criterion(leaf_b, other_object_tag, default_rule_profile)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[root_a.pk, course_a.pk, leaf_a.pk]).exists()
    assert CompetencyCriteriaGroup.objects.filter(pk__in=[root_b.pk, course_b.pk, leaf_b.pk]).count() == 3
    assert sorted(_ids_named_in(_cleanup_lines(caplog))) == sorted([root_a.pk, course_a.pk, leaf_a.pk])


def test_a_leaf_directly_under_the_root_removes_both_and_logs_two_lines(
    *, tag, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """A leaf with no course-level group above it removes itself and the root."""
    root = CompetencyCriteriaGroup.objects.create(tag=tag)
    leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=root)
    _make_criterion(leaf, object_tag, default_rule_profile)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        tag_object(object_tag.object_id, competency_taxonomy, [])

    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, leaf.pk]).exists()
    assert len(_cleanup_lines(caplog)) == 2
    assert sorted(_ids_named_in(_cleanup_lines(caplog))) == sorted([root.pk, leaf.pk])


def test_deleting_a_locked_object_tag_archives_it_and_its_criterion_and_groups(
    *, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """A locked ObjectTag is archived by the delete, which archives its criterion and groups instead of deleting."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    lock_object_tags_for_deletion(object_tag.pk)
    trigger = f"ObjectTags [{object_tag.pk}] archived"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    stored = ObjectTag.objects.get(pk=object_tag.pk)
    assert stored.archived is True
    assert CompetencyCriterion.objects.filter(pk=criterion.pk, object_tag=stored, archived=True).exists()
    assert [_archived(g) for g in (root, course_level, leaf)] == [True, True, True]
    assert _cleanup_lines(caplog) == [
        f"Group cleanup archived CompetencyCriteriaGroup {leaf.pk} after {trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {course_level.pk} after {trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {root.pk} after {trigger}",
    ]


def _delete_criterion_through_the_api(criterion, user):
    """Delete the criterion the way the authoring API does, which walks the groups itself."""
    delete_or_archive_competency_criterion(criterion.id, user)


def _delete_criterion_instance(criterion, _user):
    """Delete the criterion directly."""
    criterion.delete()


def _delete_group_instance(criterion, _user):
    """Delete the criterion's group, which cascades to the criterion."""
    criterion.group.delete()


def _delete_tag(criterion, _user):
    """Delete the competency Tag, which cascades to its groups and criteria."""
    criterion.group.tag.delete()


def _delete_taxonomy(criterion, _user):
    """Delete the taxonomy, which cascades to its tags and criteria groups and nulls the object tags' taxonomy."""
    criterion.object_tag.taxonomy.delete()


@pytest.mark.parametrize(
    "make_delete",
    [
        _delete_criterion_through_the_api,
        _delete_criterion_instance,
        _delete_group_instance,
        _delete_tag,
        _delete_taxonomy,
    ],
)
def test_other_origins_never_run_the_walk_or_log(
    make_delete: Callable, *, user, tag, course_run, object_tag, default_rule_profile, caplog, monkeypatch,
) -> None:
    """A criterion that goes away for any reason but its ObjectTag's deletion never makes the receiver walk or log."""
    _, _, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)

    def _walk_must_not_run(*_args, **_kwargs):
        raise AssertionError("the group walk ran for an ignored origin")

    monkeypatch.setattr(signal_handlers, "_cascade_remove_from_group", _walk_must_not_run)
    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        make_delete(criterion, user)

    # The receiver would turn the AssertionError into a warning line, so no line means the walk never ran.
    assert not CompetencyCriterion.objects.filter(pk=criterion.pk).exists()
    assert _cleanup_lines(caplog) == []


OTHER_OBJECT_ID = "block-v1:Org1+Python100+Fall2026+problem+p2"


def _archived(group: CompetencyCriteriaGroup) -> bool:
    """Return the group's archived flag as stored."""
    return CompetencyCriteriaGroup.objects.get(pk=group.pk).archived


def test_a_leaf_that_also_holds_an_archived_criterion_is_archived_not_removed(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """A leaf still holding an archived criterion is archived; a live sibling leaf keeps the ancestors unchanged."""
    root, course_level, leaf = _tree(tag, course_run)
    other_leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=course_level)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    archived_tag = _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID)
    archived_criterion = _make_criterion(leaf, archived_tag, default_rule_profile)
    archived_criterion.archived = True
    archived_criterion.save(update_fields=["archived"])
    live_tag = _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID + "2")
    _make_criterion(other_leaf, live_tag, default_rule_profile)
    trigger = f"CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert _archived(leaf) is True
    assert [_archived(g) for g in (course_level, root, other_leaf)] == [False, False, False]
    assert _cleanup_lines(caplog) == [f"Group cleanup archived CompetencyCriteriaGroup {leaf.pk} after {trigger}"]


def test_a_leaf_with_its_own_group_status_is_archived(
    *, user, now, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """A leaf with a learner group status row is archived instead of removed; a live sibling leaf protects the rest."""
    root, course_level, leaf = _tree(tag, course_run)
    other_leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=course_level)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    _make_criterion(other_leaf, _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID), default_rule_profile)
    StudentCompetencyCriteriaGroupStatus.objects.create(
        user=user, group=leaf, status_id=MasteryStatus.DEMONSTRATED, created=now, modified=now,
    )
    trigger = f"CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert _archived(leaf) is True
    assert [_archived(g) for g in (course_level, root, other_leaf)] == [False, False, False]
    assert _cleanup_lines(caplog) == [f"Group cleanup archived CompetencyCriteriaGroup {leaf.pk} after {trigger}"]


def test_a_course_group_left_with_only_archived_leaves_is_archived_too(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """A course-level group whose remaining leaves are all archived is archived; another course keeps the root."""
    root, course_level, leaf_y = _tree(tag, course_run)
    leaf_x = CompetencyCriteriaGroup.objects.create(tag=tag, parent=course_level, archived=True)
    criterion = _make_criterion(leaf_y, object_tag, default_rule_profile)
    archived_criterion = _make_criterion(
        leaf_x, _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID), default_rule_profile,
    )
    archived_criterion.archived = True
    archived_criterion.save(update_fields=["archived"])
    spring_run = CourseRun.objects.create(catalog_course=course_run.catalog_course, run_code="Spring2027")
    spring_level = CompetencyCriteriaGroup.objects.create(tag=tag, parent=root, course=spring_run)
    spring_leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=spring_level)
    _make_criterion(
        spring_leaf, _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID + "2"), default_rule_profile,
    )
    trigger = f"CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert not CompetencyCriteriaGroup.objects.filter(pk=leaf_y.pk).exists()
    assert _archived(course_level) is True
    assert [_archived(g) for g in (root, spring_level, spring_leaf)] == [False, False, False]
    assert _cleanup_lines(caplog) == [
        f"Group cleanup removed CompetencyCriteriaGroup {leaf_y.pk} after {trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {course_level.pk} after {trigger}",
    ]


def test_an_archived_leaf_losing_its_last_criterion_is_removed_and_logged_as_removed(
    *, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """An already-archived leaf with no status row is deleted, and so is every group above it left empty."""
    root, course_level, leaf = _tree(tag, course_run)
    leaf.archived = True
    leaf.save(update_fields=["archived"])
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    trigger = f"CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf.pk]).exists()
    assert _cleanup_lines(caplog) == [
        f"Group cleanup removed CompetencyCriteriaGroup {leaf.pk} after {trigger}",
        f"Group cleanup removed CompetencyCriteriaGroup {course_level.pk} after {trigger}",
        f"Group cleanup removed CompetencyCriteriaGroup {root.pk} after {trigger}",
    ]


def test_archiving_the_last_live_leaf_archives_every_ancestor_up_to_the_root(
    *, user, now, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """With no live group left anywhere beneath them, the course-level group and the root are archived too."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    StudentCompetencyCriteriaGroupStatus.objects.create(
        user=user, group=leaf, status_id=MasteryStatus.DEMONSTRATED, created=now, modified=now,
    )
    trigger = f"CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert [_archived(g) for g in (leaf, course_level, root)] == [True, True, True]
    assert _cleanup_lines(caplog) == [
        f"Group cleanup archived CompetencyCriteriaGroup {leaf.pk} after {trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {course_level.pk} after {trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {root.pk} after {trigger}",
    ]


def test_a_helper_failure_leaves_the_groups_warns_and_lets_the_delete_finish(
    *, tag, course_run, object_tag, default_rule_profile, monkeypatch, caplog,
) -> None:
    """A failure inside the walk keeps every group, logs one warning with the traceback, and the delete completes."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(signal_handlers, "_cascade_remove_from_group", _boom)

    with caplog.at_level(logging.WARNING, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert not CompetencyCriterion.objects.filter(pk=criterion.pk).exists()
    assert CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf.pk]).count() == 3
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        f"Group cleanup failed after CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}; "
        f"CompetencyCriteriaGroups left in place: {sorted([leaf.pk, course_level.pk, root.pk])}"
    )
    assert record.exc_info is not None


def test_a_lock_error_is_not_swallowed(
    *, tag, course_run, object_tag, default_rule_profile, monkeypatch,
) -> None:
    """A database OperationalError, such as a MySQL deadlock, propagates out of the delete."""
    _, _, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)

    def _deadlock(*_args, **_kwargs):
        raise OperationalError("Deadlock found")

    monkeypatch.setattr(signal_handlers, "_cascade_remove_from_group", _deadlock)

    with pytest.raises(OperationalError):
        delete_object_tags(object_tag.object_id)


def test_a_failure_after_removing_the_leaf_rolls_the_removal_back(
    *, tag, course_run, object_tag, default_rule_profile, monkeypatch, caplog,
) -> None:
    """A failure after the leaf was deleted undoes that deletion, so the leaf stays in place, empty."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)

    def _remove_the_leaf_then_fail(group, _user):
        CompetencyCriteriaGroup.objects.filter(pk=group.pk).delete()
        raise RuntimeError("boom")

    monkeypatch.setattr(signal_handlers, "_cascade_remove_from_group", _remove_the_leaf_then_fail)

    with caplog.at_level(logging.WARNING, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    assert CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf.pk]).count() == 3
    assert not CompetencyCriterion.objects.filter(pk=criterion.pk).exists()
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        f"Group cleanup failed after CompetencyCriterion {criterion.pk} deleted with ObjectTag {object_tag.pk}; "
        f"CompetencyCriteriaGroups left in place: {sorted([leaf.pk, course_level.pk, root.pk])}"
    )


def _second_run(course_run: CourseRun) -> CourseRun:
    """Create another run of the same catalog course."""
    return CourseRun.objects.create(catalog_course=course_run.catalog_course, run_code="Spring2027")


@pytest.mark.parametrize(
    "delete",
    [
        lambda run: run.delete(),
        lambda run: CourseRun.objects.filter(pk=run.pk).delete(),
    ],
    ids=["instance", "queryset"],
)
def test_deleting_the_only_course_under_a_root_removes_the_root_and_its_object_tags(
    delete: Callable, *, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """The root left with no course below it is removed, and so is the ObjectTag no criterion uses any more."""
    root, course_level, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    trigger = f"CompetencyCriteriaGroup {course_level.pk} deleted with CourseRun {course_run.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete(course_run)

    assert not CompetencyCriteriaGroup.objects.filter(pk=root.pk).exists()
    assert not ObjectTag.objects.filter(pk=object_tag.pk).exists()
    assert _cleanup_lines(caplog) == [f"Group cleanup removed CompetencyCriteriaGroup {root.pk} after {trigger}"]


def test_the_root_survives_unlogged_when_another_course_still_has_groups(
    *, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """A second course's groups under the same root keep the root in place and unarchived."""
    root, _, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    spring_level = CompetencyCriteriaGroup.objects.create(tag=tag, parent=root, course=_second_run(course_run))
    CompetencyCriteriaGroup.objects.create(tag=tag, parent=spring_level)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        course_run.delete()

    assert CompetencyCriteriaGroup.objects.filter(pk=root.pk).exists()
    assert _archived(root) is False
    assert _cleanup_lines(caplog) == []


def test_the_root_is_archived_when_it_has_its_own_status_row(
    *, user, now, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """A root with a learner group status row is archived, not removed, once no course is left under it."""
    root, course_level, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    StudentCompetencyCriteriaGroupStatus.objects.create(
        user=user, group=root, status_id=MasteryStatus.DEMONSTRATED, created=now, modified=now,
    )
    trigger = f"CompetencyCriteriaGroup {course_level.pk} deleted with CourseRun {course_run.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        course_run.delete()

    assert _archived(root) is True
    assert _cleanup_lines(caplog) == [f"Group cleanup archived CompetencyCriteriaGroup {root.pk} after {trigger}"]


def test_an_object_tag_another_course_still_uses_is_kept(
    *, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """An ObjectTag that a criterion in another course still references survives the deletion of this course."""
    root, _, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    spring_level = CompetencyCriteriaGroup.objects.create(tag=tag, parent=root, course=_second_run(course_run))
    spring_leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=spring_level)
    spring_criterion = _make_criterion(spring_leaf, object_tag, default_rule_profile)

    course_run.delete()

    assert ObjectTag.objects.filter(pk=object_tag.pk).exists()
    assert CompetencyCriterion.objects.filter(pk=spring_criterion.pk).exists()


def test_an_object_tag_only_an_archived_criterion_elsewhere_uses_is_locked_and_archived(
    *, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """Deleting this course archives the ObjectTag that only another course's archived criterion still uses."""
    root, _, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    spring_level = CompetencyCriteriaGroup.objects.create(
        tag=tag, parent=root, course=_second_run(course_run), archived=True,
    )
    spring_leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=spring_level, archived=True)
    spring_criterion = _make_criterion(spring_leaf, object_tag, default_rule_profile)
    spring_criterion.archived = True
    spring_criterion.save(update_fields=["archived"])

    course_run.delete()

    stored = ObjectTag.objects.get(pk=object_tag.pk)
    assert stored.archived is True
    assert stored.deletion_locked is True
    assert CompetencyCriterion.objects.filter(pk=spring_criterion.pk, object_tag=stored, archived=True).exists()
    assert CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, spring_level.pk, spring_leaf.pk]).count() == 3
    assert _archived(root) is True


def test_an_already_locked_object_tag_nothing_uses_is_archived_by_a_course_deletion(
    *, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """A locked ObjectTag that no criterion uses after the course is gone is archived, not deleted."""
    _, _, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    lock_object_tags_for_deletion(object_tag.pk)

    course_run.delete()

    stored = ObjectTag.objects.get(pk=object_tag.pk)
    assert stored.archived is True
    assert stored.deletion_locked is True


def test_one_delete_archives_a_locked_object_tag_and_removes_an_unlocked_one_with_its_groups(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """A single delete archives the locked ObjectTag's rule, and removes the unlocked one's criterion and groups."""
    tag_b = Tag.objects.create(taxonomy=competency_taxonomy, value="Revising Poetry")
    root_a, course_a, leaf_a = _tree(tag, course_run)
    root_b, course_b, leaf_b = _tree(tag_b, course_run)
    object_tag_b = _make_object_tag(competency_taxonomy, tag_b, object_tag.object_id)
    criterion_a = _make_criterion(leaf_a, object_tag, default_rule_profile)
    criterion_b = _make_criterion(leaf_b, object_tag_b, default_rule_profile)
    lock_object_tags_for_deletion(object_tag.pk)
    trigger = f"CompetencyCriterion {criterion_b.pk} deleted with ObjectTag {object_tag_b.pk}"
    archive_trigger = f"ObjectTags [{object_tag.pk}] archived"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)

    stored = ObjectTag.objects.get(pk=object_tag.pk)
    assert stored.archived is True
    assert CompetencyCriterion.objects.filter(pk=criterion_a.pk, object_tag=stored, archived=True).exists()
    assert [_archived(g) for g in (root_a, course_a, leaf_a)] == [True, True, True]
    assert not ObjectTag.objects.filter(pk=object_tag_b.pk).exists()
    assert not CompetencyCriterion.objects.filter(pk=criterion_b.pk).exists()
    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[root_b.pk, course_b.pk, leaf_b.pk]).exists()
    assert _cleanup_lines(caplog) == [
        f"Group cleanup removed CompetencyCriteriaGroup {leaf_b.pk} after {trigger}",
        f"Group cleanup removed CompetencyCriteriaGroup {course_b.pk} after {trigger}",
        f"Group cleanup removed CompetencyCriteriaGroup {root_b.pk} after {trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {leaf_a.pk} after {archive_trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {course_a.pk} after {archive_trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {root_a.pk} after {archive_trigger}",
    ]


def test_a_course_with_learner_status_cannot_be_deleted_and_logs_nothing(
    *, user, now, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """Learner status on a criterion blocks the course deletion, and the receivers never run."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    StudentCompetencyCriterionStatus.objects.create(
        user=user, criterion=criterion, status_id=MasteryStatus.DEMONSTRATED, created=now, modified=now,
    )

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        with pytest.raises(ProtectedError):
            course_run.delete()

    assert CourseRun.objects.filter(pk=course_run.pk).exists()
    assert CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf.pk]).count() == 3
    assert CompetencyCriterion.objects.filter(pk=criterion.pk).exists()
    assert ObjectTag.objects.filter(pk=object_tag.pk).exists()
    assert _cleanup_lines(caplog) == []


def test_two_runs_sharing_a_root_deleted_in_one_queryset_remove_the_root_once(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """The second course-level group finds the root already gone, logs nothing, and still reconciles its tags."""
    spring_run = _second_run(course_run)
    root, _, leaf = _tree(tag, course_run)
    spring_level = CompetencyCriteriaGroup.objects.create(tag=tag, parent=root, course=spring_run)
    spring_leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=spring_level)
    spring_object_tag = _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID)
    _make_criterion(leaf, object_tag, default_rule_profile)
    _make_criterion(spring_leaf, spring_object_tag, default_rule_profile)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        CourseRun.objects.filter(pk__in=[course_run.pk, spring_run.pk]).delete()

    assert not CompetencyCriteriaGroup.objects.filter(pk=root.pk).exists()
    assert not ObjectTag.objects.filter(pk__in=[object_tag.pk, spring_object_tag.pk]).exists()
    lines = _cleanup_lines(caplog)
    assert len(lines) == 1
    assert lines[0].startswith(f"Group cleanup removed CompetencyCriteriaGroup {root.pk} after ")


def test_a_course_group_with_no_criteria_still_cleans_up_the_root(*, tag, course_run, caplog) -> None:
    """A course-level group with no criteria anywhere beneath it still lets the root be removed."""
    root, course_level, _ = _tree(tag, course_run)
    trigger = f"CompetencyCriteriaGroup {course_level.pk} deleted with CourseRun {course_run.pk}"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        course_run.delete()

    assert not CompetencyCriteriaGroup.objects.filter(pk=root.pk).exists()
    assert _cleanup_lines(caplog) == [f"Group cleanup removed CompetencyCriteriaGroup {root.pk} after {trigger}"]


def test_both_orders_of_a_course_deletion_end_with_the_same_rows(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """Removing the ObjectTag first, or the course first, leaves no group and no ObjectTag, and warns about nothing."""
    tag_b = Tag.objects.create(taxonomy=competency_taxonomy, value="Revising Poetry")
    run_b = _second_run(course_run)
    object_tag_b = _make_object_tag(competency_taxonomy, tag_b, OTHER_OBJECT_ID)
    root_a, course_a, leaf_a = _tree(tag, course_run)
    root_b, _, leaf_b = _tree(tag_b, run_b)
    _make_criterion(leaf_a, object_tag, default_rule_profile)
    _make_criterion(leaf_b, object_tag_b, default_rule_profile)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(object_tag.object_id)
        course_run.delete()
    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert sorted(_ids_named_in(_cleanup_lines(caplog))) == sorted([leaf_a.pk, course_a.pk, root_a.pk])

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        run_b.delete()
    assert [r.levelno for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert _ids_named_in(_cleanup_lines(caplog)) == [root_b.pk]

    assert not CompetencyCriteriaGroup.objects.filter(tag__in=[tag, tag_b]).exists()
    assert not ObjectTag.objects.filter(pk__in=[object_tag.pk, object_tag_b.pk]).exists()


def test_a_failed_tag_reconcile_warns_keeps_the_object_tags_and_lets_the_delete_finish(
    *, tag, course_run, object_tag, default_rule_profile, monkeypatch, caplog,
) -> None:
    """The ObjectTag step failing never undoes the course deletion or the root removal."""
    root, _, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    course_run_pk = course_run.pk

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(signal_handlers, "_reconcile_object_tags", _boom)

    with caplog.at_level(logging.WARNING, logger=signal_handlers.__name__):
        course_run.delete()

    assert not CourseRun.objects.filter(pk=course_run_pk).exists()
    assert not CompetencyCriteriaGroup.objects.filter(pk=root.pk).exists()
    assert ObjectTag.objects.filter(pk=object_tag.pk).exists()
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage().endswith(f"; ObjectTags left in place: [{object_tag.pk}]")
    assert record.exc_info is not None


def test_a_helper_failure_on_a_course_run_delete_warns_and_lets_the_delete_finish(
    *, tag, course_run, object_tag, default_rule_profile, monkeypatch, caplog,
) -> None:
    """A failing walk keeps the root, the course delete cascades as usual, and the ObjectTag cleanup still runs."""
    root, course_level, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)
    course_run_pk = course_run.pk

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(signal_handlers, "_cascade_remove_from_group", _boom)

    with caplog.at_level(logging.WARNING, logger=signal_handlers.__name__):
        course_run.delete()

    assert not CourseRun.objects.filter(pk=course_run_pk).exists()
    assert CompetencyCriteriaGroup.objects.filter(pk=root.pk).exists()
    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[course_level.pk, leaf.pk]).exists()
    assert not ObjectTag.objects.filter(pk=object_tag.pk).exists()
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        f"Group cleanup failed after CompetencyCriteriaGroup {course_level.pk} deleted with CourseRun "
        f"{course_run_pk}; CompetencyCriteriaGroups left in place: [{root.pk}]"
    )
    assert record.exc_info is not None


def test_a_lock_error_in_the_tag_cleanup_is_not_swallowed(
    *, tag, course_run, object_tag, default_rule_profile, monkeypatch,
) -> None:
    """A database OperationalError raised by the ObjectTag cleanup propagates out of the course delete."""
    _, _, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)

    def _deadlock(*_args, **_kwargs):
        raise OperationalError("Deadlock found")

    monkeypatch.setattr(signal_handlers, "_reconcile_object_tags", _deadlock)

    with pytest.raises(OperationalError):
        course_run.delete()


def test_a_course_group_with_no_parent_still_reconciles_its_object_tags(
    *, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """With no root above it there is nothing to walk, but the ObjectTags its criteria used are still removed."""
    course_level = CompetencyCriteriaGroup.objects.create(tag=tag, course=course_run)
    leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=course_level)
    _make_criterion(leaf, object_tag, default_rule_profile)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        course_run.delete()

    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[course_level.pk, leaf.pk]).exists()
    assert not ObjectTag.objects.filter(pk=object_tag.pk).exists()
    assert _cleanup_lines(caplog) == []


def _archive_locked_association(object_tag: ObjectTag, taxonomy: CompetencyTaxonomy) -> None:
    """Lock ``object_tag`` and remove it with tag_object(), which archives it instead of deleting it."""
    lock_object_tags_for_deletion(object_tag.pk)
    tag_object(object_tag.object_id, taxonomy, [])


def _criterion_archived(criterion: CompetencyCriterion) -> bool:
    """Return the criterion's archived flag as stored."""
    return CompetencyCriterion.objects.get(pk=criterion.pk).archived


def _history_count(model_name: str, ids: list[int]) -> int:
    """
    Return how many history rows the given ``CompetencyCriterion`` or ``CompetencyCriteriaGroup`` ids have.

    The historical model comes from the app registry because simple_history's ``.history`` has no type stubs.
    """
    return apps.get_model("openedx_learning", f"Historical{model_name}").objects.filter(id__in=ids).count()


def test_archiving_a_locked_association_archives_its_criterion_and_empty_lineage(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """Removing a locked association archives its criterion and each group above it left with no live child."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    trigger = f"ObjectTags [{object_tag.pk}] archived"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        _archive_locked_association(object_tag, competency_taxonomy)

    assert ObjectTag.objects.get(pk=object_tag.pk).archived is True
    assert CompetencyCriterion.objects.get(pk=criterion.pk).archived is True
    assert [_archived(g) for g in (leaf, course_level, root)] == [True, True, True]
    assert _cleanup_lines(caplog) == [
        f"Group cleanup archived CompetencyCriteriaGroup {leaf.pk} after {trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {course_level.pk} after {trigger}",
        f"Group cleanup archived CompetencyCriteriaGroup {root.pk} after {trigger}",
    ]


def test_two_criteria_in_one_group_log_the_group_once(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """Both criteria are archived, and the group they share is logged once, not once per criterion."""
    _, _, leaf = _tree(tag, course_run)
    first = _make_criterion(leaf, object_tag, default_rule_profile)
    other_tag = _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID)
    second = _make_criterion(leaf, other_tag, default_rule_profile)
    lock_object_tags_for_deletion([object_tag.pk, other_tag.pk])

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        ObjectTag.objects.filter(pk__in=[object_tag.pk, other_tag.pk]).delete()

    assert CompetencyCriterion.objects.filter(pk__in=[first.pk, second.pk], archived=True).count() == 2
    assert _ids_named_in(_cleanup_lines(caplog)).count(leaf.pk) == 1


def test_reviving_an_association_leaves_the_rule_archived(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """Re-applying the tag revives the association, while the criterion and its groups stay archived, silently."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    _archive_locked_association(object_tag, competency_taxonomy)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        tag_object(object_tag.object_id, competency_taxonomy, [tag.value])

    assert ObjectTag.objects.get(pk=object_tag.pk).archived is False
    assert CompetencyCriterion.objects.get(pk=criterion.pk).archived is True
    assert [_archived(g) for g in (leaf, course_level, root)] == [True, True, True]
    assert _cleanup_lines(caplog) == []


@pytest.mark.parametrize("sibling_group", ["same_leaf", "sibling_leaf"])
def test_live_sibling_criterion_stops_the_archive_walk(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog, sibling_group,
) -> None:
    """
    A live criterion next to the archived one keeps its own group and every group above it live.

    A sibling in the same leaf keeps that leaf live, and a sibling in another leaf keeps the course-level group
    and the root live. Only the groups the archive emptied are logged.
    """
    root, course_level, leaf = _tree(tag, course_run)
    sibling_leaf = leaf if sibling_group == "same_leaf" else CompetencyCriteriaGroup.objects.create(
        tag=tag, parent=course_level
    )
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    sibling = _make_criterion(
        sibling_leaf, _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID), default_rule_profile
    )
    trigger = f"ObjectTags [{object_tag.pk}] archived"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        _archive_locked_association(object_tag, competency_taxonomy)

    leaf_is_archived = sibling_group == "sibling_leaf"
    assert CompetencyCriterion.objects.get(pk=criterion.pk).archived is True
    assert CompetencyCriterion.objects.get(pk=sibling.pk).archived is False
    assert [_archived(g) for g in (leaf, course_level, root, sibling_leaf)] == [
        leaf_is_archived, False, False, False
    ]
    assert _cleanup_lines(caplog) == (
        [f"Group cleanup archived CompetencyCriteriaGroup {leaf.pk} after {trigger}"] if leaf_is_archived else []
    )


def test_archiving_the_association_of_an_already_archived_rule_is_a_silent_no_op(
    *, user, now, tag, course_run, object_tag, default_rule_profile, caplog, django_assert_num_queries,
) -> None:
    """
    Archiving a criterion through the authoring API archives its association, and that signal changes nothing more.

    The receiver is reached, finds no live criterion, and writes no further history row. Sent again, it runs one
    query, so it takes no group lock.
    """
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    StudentCompetencyCriterionStatus.objects.create(
        user=user, criterion=criterion, status_id=MasteryStatus.DEMONSTRATED, created=now, modified=now,
    )
    group_ids = [leaf.pk, course_level.pk, root.pk]
    criterion_history = _history_count("CompetencyCriterion", [criterion.pk])
    group_history = _history_count("CompetencyCriteriaGroup", group_ids)
    sent: list[list[int]] = []

    def _record(object_tag_ids, **_kwargs):
        sent.append(object_tag_ids)

    object_tags_archived.connect(_record)
    try:
        with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
            delete_or_archive_competency_criterion(criterion.id, user)
    finally:
        object_tags_archived.disconnect(_record)

    assert sent == [[object_tag.pk]]
    assert ObjectTag.objects.get(pk=object_tag.pk).archived is True
    assert CompetencyCriterion.objects.get(pk=criterion.pk).archived is True
    assert [_archived(g) for g in (leaf, course_level, root)] == [True, True, True]
    assert _history_count("CompetencyCriterion", [criterion.pk]) == criterion_history + 1
    assert _history_count("CompetencyCriteriaGroup", group_ids) == group_history + len(group_ids)
    assert _cleanup_lines(caplog) == []
    with django_assert_num_queries(1):
        object_tags_archived.send(sender=ObjectTag, object_tag_ids=sent[0], using="default")


def test_group_archive_reconcile_is_a_silent_no_op(
    *, user, now, tag, course_run, object_tag, default_rule_profile, caplog, django_assert_num_queries,
) -> None:
    """The signal sent from inside a group archive's own reconcile finds no live criteria and runs one query."""
    _, _, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    StudentCompetencyCriterionStatus.objects.create(
        user=user, criterion=criterion, status_id=MasteryStatus.DEMONSTRATED, created=now, modified=now,
    )
    sent: list[list[int]] = []

    def _record(object_tag_ids, **_kwargs):
        sent.append(object_tag_ids)

    object_tags_archived.connect(_record)
    try:
        with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
            delete_or_archive_competency_criteria_group(leaf.pk, user)
    finally:
        object_tags_archived.disconnect(_record)

    assert CompetencyCriterion.objects.get(pk=criterion.pk).archived is True
    assert sent == [[object_tag.pk]]
    assert _cleanup_lines(caplog) == []
    with django_assert_num_queries(1):
        object_tags_archived.send(sender=ObjectTag, object_tag_ids=sent[0], using="default")


def test_object_tags_archived_twice_changes_and_logs_nothing_the_second_time(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, caplog,
) -> None:
    """Sending the same ids again, as a concurrent archive can, writes no history row and logs nothing."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    _archive_locked_association(object_tag, competency_taxonomy)
    group_ids = [leaf.pk, course_level.pk, root.pk]
    criterion_history = _history_count("CompetencyCriterion", [criterion.pk])
    group_history = _history_count("CompetencyCriteriaGroup", group_ids)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        object_tags_archived.send(sender=ObjectTag, object_tag_ids=[object_tag.pk])

    assert _history_count("CompetencyCriterion", [criterion.pk]) == criterion_history
    assert _history_count("CompetencyCriteriaGroup", group_ids) == group_history
    assert _cleanup_lines(caplog) == []


def test_object_tags_archived_for_a_live_association_changes_and_logs_nothing(
    *, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """A signal that names an association that is not archived leaves its live criterion and groups alone."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    group_ids = [leaf.pk, course_level.pk, root.pk]
    group_history = _history_count("CompetencyCriteriaGroup", group_ids)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        object_tags_archived.send(sender=ObjectTag, object_tag_ids=[object_tag.pk], using="default")

    assert _criterion_archived(criterion) is False
    assert [_archived(g) for g in (leaf, course_level, root)] == [False, False, False]
    assert _history_count("CompetencyCriteriaGroup", group_ids) == group_history
    assert _cleanup_lines(caplog) == []


def test_association_archive_cleans_up_groups_in_tag_then_group_order(
    *, competency_taxonomy, tag, course_run, default_rule_profile, caplog,
) -> None:
    """
    Groups are processed by competency tag, then by group, so every receiver takes the Tag locks in one order.

    The second competency's groups and association are created first, so they have the lower primary keys, while
    its tag has the higher one. Only the receiver's ordering puts the first competency's groups first.
    """
    tag_b = Tag.objects.create(taxonomy=competency_taxonomy, value="Other Competency")
    groups_b = _tree(tag_b, course_run)
    object_tag_b = _make_object_tag(competency_taxonomy, tag_b, OTHER_OBJECT_ID)
    groups_a = _tree(tag, course_run)
    object_tag_a = _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID)
    _make_criterion(groups_b[2], object_tag_b, default_rule_profile)
    _make_criterion(groups_a[2], object_tag_a, default_rule_profile)
    lock_object_tags_for_deletion([object_tag_a.pk, object_tag_b.pk])

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_object_tags(OTHER_OBJECT_ID)

    logged_ids = _ids_named_in(_cleanup_lines(caplog))
    assert len(logged_ids) == 6
    assert set(logged_ids[:3]) == {g.pk for g in groups_a}
    assert set(logged_ids[3:]) == {g.pk for g in groups_b}


def test_association_archive_cleanup_failure_is_logged_and_criteria_stay_unarchived(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, monkeypatch, caplog,
) -> None:
    """A failing cascade rolls back that group's savepoint, including the criterion archive; the archive commits."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(signal_handlers, "_cascade_remove_from_group", _boom)

    with caplog.at_level(logging.WARNING, logger=signal_handlers.__name__):
        _archive_locked_association(object_tag, competency_taxonomy)

    assert ObjectTag.objects.get(pk=object_tag.pk).archived is True
    assert CompetencyCriterion.objects.get(pk=criterion.pk).archived is False
    assert [_archived(g) for g in (leaf, course_level, root)] == [False, False, False]
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        f"Group cleanup failed after ObjectTags [{object_tag.pk}] archived; "
        f"CompetencyCriteriaGroups left in place: {sorted([leaf.pk, course_level.pk, root.pk])}"
    )
    assert record.exc_info is not None


def test_a_lock_error_in_the_association_archive_cleanup_is_not_swallowed(
    *, tag, course_run, competency_taxonomy, object_tag, default_rule_profile, monkeypatch,
) -> None:
    """A database OperationalError propagates out of the archive that triggered the receiver."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)

    def _deadlock(*_args, **_kwargs):
        raise OperationalError("Deadlock found")

    monkeypatch.setattr(signal_handlers, "_cascade_remove_from_group", _deadlock)

    with pytest.raises(OperationalError):
        _archive_locked_association(object_tag, competency_taxonomy)

    assert ObjectTag.objects.get(pk=object_tag.pk).archived is False
    assert CompetencyCriterion.objects.get(pk=criterion.pk).archived is False
    assert [_archived(g) for g in (root, course_level, leaf)] == [False, False, False]


def _archive_tags_through_the_taxonomy(taxonomy: CompetencyTaxonomy, *tags: Tag) -> None:
    """Archive ``tags``, each of which must have a deletion-locked association."""
    taxonomy.delete_tags([t.value for t in tags])


def test_archiving_a_tag_archives_its_competencys_whole_tree(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """
    Every live group and criterion of the competency, in both courses, is archived, one line per group.

    Members that were already archived are not written again, and nothing else moves.
    """
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    second_course_level = CompetencyCriteriaGroup.objects.create(tag=tag, parent=root, course=_second_run(course_run))
    second_leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=second_course_level)
    second_criterion = _make_criterion(
        second_leaf, _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID + "2"), default_rule_profile
    )
    done_leaf = CompetencyCriteriaGroup.objects.create(tag=tag, parent=course_level, archived=True)
    done_criterion = CompetencyCriterion.objects.create(
        group=done_leaf,
        object_tag=_make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID),
        rule_profile=default_rule_profile,
        archived=True,
    )
    other_tag = Tag.objects.create(taxonomy=competency_taxonomy, value="Other Competency")
    other_groups = _tree(other_tag, course_run)
    other_criterion = _make_criterion(
        other_groups[2],
        _make_object_tag(competency_taxonomy, other_tag, "block-v1:Org1+Python100+Fall2026+problem+p3"),
        default_rule_profile,
    )
    lock_object_tags_for_deletion(object_tag.pk)
    done_group_history = _history_count("CompetencyCriteriaGroup", [done_leaf.pk])
    done_criterion_history = _history_count("CompetencyCriterion", [done_criterion.pk])
    archived_groups = [root, course_level, leaf, second_course_level, second_leaf]

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        _archive_tags_through_the_taxonomy(competency_taxonomy, tag)

    assert Tag.objects.get(pk=tag.pk).archived is True
    assert [_archived(g) for g in archived_groups] == [True] * 5
    assert _criterion_archived(criterion) is True
    assert _criterion_archived(second_criterion) is True
    assert _cleanup_lines(caplog) == [
        f"Group cleanup archived CompetencyCriteriaGroup {pk} after Tag {tag.pk} archived"
        for pk in sorted(g.pk for g in archived_groups)
    ]
    assert _history_count("CompetencyCriteriaGroup", [done_leaf.pk]) == done_group_history
    assert _history_count("CompetencyCriterion", [done_criterion.pk]) == done_criterion_history
    assert [_archived(g) for g in other_groups] == [False, False, False]
    assert _criterion_archived(other_criterion) is False


def test_tag_archive_reconciles_the_competencys_associations(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """An association only archived criteria use ends locked and archived, so a content delete cannot cascade."""
    _, _, leaf = _tree(tag, course_run)
    first = _make_criterion(leaf, object_tag, default_rule_profile)
    unlocked = _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID)
    second = _make_criterion(leaf, unlocked, default_rule_profile)
    lock_object_tags_for_deletion(object_tag.pk)

    _archive_tags_through_the_taxonomy(competency_taxonomy, tag)

    stored = ObjectTag.objects.get(pk=unlocked.pk)
    assert (stored.deletion_locked, stored.archived) == (True, True)
    delete_object_tags(OTHER_OBJECT_ID)
    assert CompetencyCriterion.objects.filter(pk__in=[first.pk, second.pk], archived=True).count() == 2


def test_tag_archive_leaves_an_association_revived_after_its_rule_was_archived(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """
    Archiving the tag later does not archive again an association that re-tagging revived.

    The rule was archived with its association, and tagging the object again revived the association. Only the
    rules the tag archive writes have their associations settled, so the revived one stays live.
    """
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    _archive_locked_association(object_tag, competency_taxonomy)
    tag_object(object_tag.object_id, competency_taxonomy, [tag.value])
    assert ObjectTag.objects.get(pk=object_tag.pk).archived is False

    _archive_tags_through_the_taxonomy(competency_taxonomy, tag)

    stored = ObjectTag.objects.get(pk=object_tag.pk)
    assert (stored.archived, stored.deletion_locked) == (False, True)
    assert Tag.objects.get(pk=tag.pk).archived is True
    assert _criterion_archived(criterion) is True
    assert [_archived(g) for g in (leaf, course_level, root)] == [True, True, True]


def test_import_replace_omitting_a_locked_competency_tag_archives_its_tree(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """A replace import that omits a competency tag with a locked association archives the competency's tree."""
    tag.external_id = "COMP-A"
    tag.save()
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    lock_object_tags_for_deletion(object_tag.pk)
    file = BytesIO(json.dumps({"tags": [{"id": "COMP-B", "value": "Beta"}]}).encode())

    result, _task, _plan = import_tags(competency_taxonomy, file, ParserFormat.JSON, replace=True)

    assert result
    assert Tag.objects.get(pk=tag.pk).archived is True
    assert [_archived(g) for g in (root, course_level, leaf)] == [True, True, True]
    assert _criterion_archived(criterion) is True


def test_import_replace_omitting_an_unlocked_competency_tag_removes_its_tree_silently(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """With no learner progress, a replace import that omits a competency tag leaves it no rules or groups, unlogged."""
    tag.external_id = "COMP-A"
    tag.save()
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    file = BytesIO(json.dumps({"tags": [{"id": "COMP-B", "value": "Beta"}]}).encode())

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        result, _task, _plan = import_tags(competency_taxonomy, file, ParserFormat.JSON, replace=True)

    assert result
    assert not Tag.objects.filter(pk=tag.pk).exists()
    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf.pk]).exists()
    assert not CompetencyCriterion.objects.filter(pk=criterion.pk).exists()
    assert _cleanup_lines(caplog) == []


def test_import_replace_omitting_a_parent_and_its_children_archives_or_removes_each_competency_tree(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """
    Child C has learner progress and child D has none, and the import omits their parent P and both of them.

    P and C stay as archived tags, with their trees archived and logged. D's tag is deleted with its tree,
    and nothing is logged for it.
    """
    tag.external_id = "COMP-P"
    tag.save()
    child_c = Tag.objects.create(taxonomy=competency_taxonomy, value="Child C", external_id="COMP-C", parent=tag)
    child_d = Tag.objects.create(taxonomy=competency_taxonomy, value="Child D", external_id="COMP-D", parent=tag)
    groups_p = _tree(tag, course_run)
    groups_c = _tree(child_c, course_run)
    groups_d = _tree(child_d, course_run)
    criterion_p = _make_criterion(groups_p[2], object_tag, default_rule_profile)
    object_tag_c = _make_object_tag(competency_taxonomy, child_c, OTHER_OBJECT_ID)
    criterion_c = _make_criterion(groups_c[2], object_tag_c, default_rule_profile)
    criterion_d = _make_criterion(
        groups_d[2], _make_object_tag(competency_taxonomy, child_d, OTHER_OBJECT_ID + "2"), default_rule_profile
    )
    lock_object_tags_for_deletion(object_tag_c.pk)
    file = BytesIO(json.dumps({"tags": [{"id": "COMP-B", "value": "Beta"}]}).encode())

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        result, _task, _plan = import_tags(competency_taxonomy, file, ParserFormat.JSON, replace=True)

    assert result
    assert [Tag.objects.get(pk=t.pk).archived for t in (tag, child_c)] == [True, True]
    assert not Tag.objects.filter(pk=child_d.pk).exists()
    assert [_archived(g) for g in (*groups_p, *groups_c)] == [True] * 6
    assert _criterion_archived(criterion_p) is True
    assert _criterion_archived(criterion_c) is True
    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[g.pk for g in groups_d]).exists()
    assert not CompetencyCriterion.objects.filter(pk=criterion_d.pk).exists()
    assert _cleanup_lines(caplog) == [
        f"Group cleanup archived CompetencyCriteriaGroup {g.pk} after Tag {archived_tag.pk} archived"
        for archived_tag, groups in ((tag, groups_p), (child_c, groups_c))
        for g in sorted(groups, key=lambda group: group.pk)
    ]


def test_archiving_a_taxonomy_archives_every_competency_tree_in_it(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile, caplog,
) -> None:
    """delete_taxonomy() archives every competency's tree in that taxonomy and leaves another taxonomy's trees live."""
    # The second competency's tree is created first, so that its groups have the lower primary keys while its tag
    # has the higher one. Only the receiver's ordering by tag then puts the first competency's groups first.
    tag_b = Tag.objects.create(taxonomy=competency_taxonomy, value="Other Competency")
    groups_b = _tree(tag_b, course_run)
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    criterion_b = _make_criterion(
        groups_b[2],
        _make_object_tag(competency_taxonomy, tag_b, "block-v1:Org1+Python100+Fall2026+problem+p3"),
        default_rule_profile,
    )
    elsewhere = CompetencyTaxonomy.objects.create(name="Teaching", export_id="teaching-v1")
    tag_elsewhere = Tag.objects.create(taxonomy=elsewhere, value="Lesson Planning")
    groups_elsewhere = _tree(tag_elsewhere, course_run)
    criterion_elsewhere = _make_criterion(
        groups_elsewhere[2],
        _make_object_tag(elsewhere, tag_elsewhere, "block-v1:Org1+Python100+Fall2026+problem+p4"),
        default_rule_profile,
    )
    lock_object_tags_for_deletion(object_tag.pk)
    group_ids_elsewhere = [g.pk for g in groups_elsewhere]
    group_history_elsewhere = _history_count("CompetencyCriteriaGroup", group_ids_elsewhere)
    trigger = f"Taxonomy {competency_taxonomy.pk} archived"

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        delete_taxonomy(competency_taxonomy)

    assert CompetencyTaxonomy.objects.get(pk=competency_taxonomy.pk).archived is True
    groups = [root, course_level, leaf, *groups_b]
    assert [_archived(g) for g in groups] == [True] * 6
    assert _criterion_archived(criterion) is True
    assert _criterion_archived(criterion_b) is True
    lines = _cleanup_lines(caplog)
    logged_ids = _ids_named_in(lines)
    assert sorted(logged_ids) == sorted(g.pk for g in groups)
    assert set(logged_ids[:3]) == {root.pk, course_level.pk, leaf.pk}
    assert set(logged_ids[3:]) == {g.pk for g in groups_b}
    assert all(line.endswith(f"after {trigger}") for line in lines)
    assert [_archived(g) for g in groups_elsewhere] == [False, False, False]
    assert _criterion_archived(criterion_elsewhere) is False
    assert _history_count("CompetencyCriteriaGroup", group_ids_elsewhere) == group_history_elsewhere


@pytest.mark.parametrize("signal", [tags_archived, taxonomies_archived], ids=["tag", "taxonomy"])
def test_tag_or_taxonomy_archived_for_live_rows_changes_and_logs_nothing(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile, caplog, signal,
) -> None:
    """A signal that names a tag or taxonomy that is not archived leaves the competency's live tree alone."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    group_ids = [leaf.pk, course_level.pk, root.pk]
    group_history = _history_count("CompetencyCriteriaGroup", group_ids)
    payload = {"tag_ids": [tag.pk]} if signal is tags_archived else {"taxonomy_ids": [competency_taxonomy.pk]}

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        signal.send(sender=Tag if signal is tags_archived else Taxonomy, using="default", **payload)

    assert _criterion_archived(criterion) is False
    assert [_archived(g) for g in (leaf, course_level, root)] == [False, False, False]
    assert _history_count("CompetencyCriteriaGroup", group_ids) == group_history
    assert _cleanup_lines(caplog) == []


def test_tag_archive_failure_rolls_back_only_that_competency(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile, caplog, monkeypatch,
) -> None:
    """A failing subtree archive undoes one competency; the other archives, the trigger commits, a warning names it."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    tag_b = Tag.objects.create(taxonomy=competency_taxonomy, value="Other Competency")
    groups_b = _tree(tag_b, course_run)
    object_tag_b = _make_object_tag(competency_taxonomy, tag_b, OTHER_OBJECT_ID)
    criterion_b = _make_criterion(groups_b[2], object_tag_b, default_rule_profile)
    lock_object_tags_for_deletion([object_tag.pk, object_tag_b.pk])
    original = _archive_competency_criteria_subtree

    def _fail_for_the_first_competency(target_group, *args):
        if target_group.tag_id == tag.pk:
            # Write something first, so the test shows the savepoint undoing it.
            target_group.archived = True
            target_group.save(update_fields=["archived"])
            raise RuntimeError("boom")
        return original(target_group, *args)

    monkeypatch.setattr(signal_handlers, "_archive_competency_criteria_subtree", _fail_for_the_first_competency)

    with caplog.at_level(logging.INFO, logger=signal_handlers.__name__):
        _archive_tags_through_the_taxonomy(competency_taxonomy, tag, tag_b)

    assert [Tag.objects.get(pk=t.pk).archived for t in (tag, tag_b)] == [True, True]
    assert [_archived(g) for g in (root, course_level, leaf)] == [False, False, False]
    assert _criterion_archived(criterion) is False
    assert [_archived(g) for g in groups_b] == [True, True, True]
    assert _criterion_archived(criterion_b) is True
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning.getMessage() == (
        f"Group cleanup failed after Tag {tag.pk} archived; "
        f"CompetencyCriteriaGroups left in place: {sorted([leaf.pk, course_level.pk, root.pk])}"
    )
    assert warning.exc_info is not None
    archived_lines = [line for line in _cleanup_lines(caplog) if line.startswith("Group cleanup archived")]
    assert sorted(_ids_named_in(archived_lines)) == sorted(g.pk for g in groups_b)


def test_a_lock_error_in_the_tag_archive_cleanup_is_not_swallowed(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile, monkeypatch,
) -> None:
    """A database OperationalError propagates out of the archive that triggered the receiver."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    lock_object_tags_for_deletion(object_tag.pk)

    def _deadlock(*_args, **_kwargs):
        raise OperationalError("Deadlock found")

    monkeypatch.setattr(signal_handlers, "_archive_competency_criteria_subtree", _deadlock)

    with pytest.raises(OperationalError):
        _archive_tags_through_the_taxonomy(competency_taxonomy, tag)

    assert Tag.objects.get(pk=tag.pk).archived is False
    assert _criterion_archived(criterion) is False
    assert [_archived(g) for g in (root, course_level, leaf)] == [False, False, False]


def test_the_criterion_receiver_runs_every_query_on_the_alias_it_was_sent(
    *, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """With reads routed to an unconfigured alias, the cleanup still completes, so no query left the sent alias."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)

    with override_settings(DATABASE_ROUTERS=[WritesToPrimaryRouter()]):
        ObjectTag.objects.filter(pk=object_tag.pk).delete()

    assert not CompetencyCriterion.objects.filter(pk=criterion.pk).exists()
    assert not CompetencyCriteriaGroup.objects.filter(pk__in=[root.pk, course_level.pk, leaf.pk]).exists()


def test_the_course_run_receivers_run_every_query_on_the_alias_they_were_sent(
    *, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """The receivers take the alias from the delete signals, not from the router, which would send reads elsewhere."""
    root, _, leaf = _tree(tag, course_run)
    _make_criterion(leaf, object_tag, default_rule_profile)

    with override_settings(DATABASE_ROUTERS=[WritesToPrimaryRouter()]):
        CourseRun.objects.filter(pk=course_run.pk).delete()

    assert not CompetencyCriteriaGroup.objects.filter(pk=root.pk).exists()
    assert not ObjectTag.objects.filter(pk=object_tag.pk).exists()


def test_the_association_archive_receiver_runs_every_query_on_the_alias_it_was_sent(
    *, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """With reads routed to an unconfigured alias, the cleanup still completes, so no query left the sent alias."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    lock_object_tags_for_deletion(object_tag.pk)

    with override_settings(DATABASE_ROUTERS=[WritesToPrimaryRouter()]):
        ObjectTag.objects.filter(pk=object_tag.pk).delete()

    assert _criterion_archived(criterion) is True
    assert [_archived(g) for g in (root, course_level, leaf)] == [True, True, True]


def test_the_tag_archive_receiver_runs_every_query_on_the_alias_it_was_sent(
    *, competency_taxonomy, tag, course_run, object_tag, default_rule_profile,
) -> None:
    """The receiver takes the alias from the signal, not from the router, which would send its reads elsewhere."""
    root, course_level, leaf = _tree(tag, course_run)
    criterion = _make_criterion(leaf, object_tag, default_rule_profile)
    unlocked = _make_object_tag(competency_taxonomy, tag, OTHER_OBJECT_ID)
    _make_criterion(leaf, unlocked, default_rule_profile)
    Tag.objects.filter(pk=tag.pk).update(archived=True)
    Taxonomy.objects.filter(pk=competency_taxonomy.pk).update(archived=True)

    with override_settings(DATABASE_ROUTERS=[WritesToPrimaryRouter()]):
        tags_archived.send(sender=Tag, tag_ids=[tag.pk], using="default")
        taxonomies_archived.send(sender=Taxonomy, taxonomy_ids=[competency_taxonomy.pk], using="default")

    assert _criterion_archived(criterion) is True
    assert [_archived(g) for g in (root, course_level, leaf)] == [True, True, True]
    stored = ObjectTag.objects.get(pk=unlocked.pk)
    assert (stored.deletion_locked, stored.archived) == (True, True)
