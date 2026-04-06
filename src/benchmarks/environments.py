"""Toy text-based environments for evaluating latent planning agents.

Each environment exposes a text-described state, a set of valid actions per step,
and a goal condition. Environments are deterministic given a seed and designed so
that an optimal agent can solve them in a known minimum number of steps, while
distractor actions inflate the branching factor.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class StepResult:
    """Return value of ``TextEnvironment.step()``."""

    state: str
    done: bool


class TextEnvironment(ABC):
    """Base class for text-based evaluation environments.

    Subclasses define a state machine where states and actions are natural
    language strings. The environment tracks how many steps the agent has taken
    and exposes the optimal (oracle) action at every state.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._step_count: int = 0
        self._done: bool = False
        self._state: str = ""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable environment name."""

    @property
    @abstractmethod
    def goal(self) -> str:
        """Natural language description of the goal state."""

    @property
    def state(self) -> str:
        """Current state description."""
        return self._state

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def done(self) -> bool:
        return self._done

    @abstractmethod
    def reset(self) -> str:
        """Reset the environment and return the initial state description."""

    @abstractmethod
    def get_valid_actions(self) -> list[str]:
        """Return all valid actions in the current state (including distractors)."""

    @abstractmethod
    def get_oracle_action(self) -> str | None:
        """Return the single optimal action in the current state, or None if done."""

    @property
    @abstractmethod
    def oracle_steps(self) -> int:
        """Minimum number of steps an optimal agent needs to reach the goal."""

    def step(self, action: str) -> StepResult:
        """Execute *action* and advance environment state.

        Args:
            action: Must be one of ``get_valid_actions()``.

        Returns:
            A ``StepResult`` with the new state description and done flag.

        Raises:
            ValueError: If action is not currently valid.
        """
        if self._done:
            raise RuntimeError("Environment is done; call reset().")
        valid = self.get_valid_actions()
        if action not in valid:
            raise ValueError(f"Invalid action: {action!r}. Valid: {valid}")
        self._step_count += 1
        result = self._execute(action)
        self._state = result.state
        self._done = result.done
        return result

    @abstractmethod
    def _execute(self, action: str) -> StepResult:
        """Internal state transition logic implemented by each environment."""


# ======================================================================
# Concrete environments
# ======================================================================


class DocumentWorkflow(TextEnvironment):
    """Manage a document through a multi-step workflow.

    Optimal path (5 steps):
        open_document -> read_contents -> summarize -> file_in_archive -> send_summary
    """

    STAGES: list[str] = [
        "inbox",
        "document_open",
        "contents_read",
        "summarized",
        "filed",
        "sent",
    ]

    OPTIMAL_PATH: list[str] = [
        "open_document",
        "read_contents",
        "summarize",
        "file_in_archive",
        "send_summary",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "inbox": [
            "delete_document",
            "forward_unread",
            "mark_as_spam",
            "print_document",
            "share_link",
        ],
        "document_open": [
            "close_document",
            "rename_file",
            "convert_to_pdf",
            "add_comment",
            "highlight_random",
        ],
        "contents_read": [
            "translate_document",
            "run_spell_check",
            "export_metadata",
            "add_watermark",
            "count_words",
        ],
        "summarized": [
            "re_summarize",
            "discard_summary",
            "export_as_csv",
            "append_footer",
            "encrypt_file",
        ],
        "filed": [
            "unarchive",
            "duplicate_file",
            "create_backup",
            "change_permissions",
            "move_to_trash",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "inbox": "A new document has arrived in your inbox. It has not been opened yet.",
        "document_open": "The document is now open on your screen. You can see the title but have not read the full contents.",
        "contents_read": "You have read the full contents of the document. Key information has been identified.",
        "summarized": "A concise summary of the document has been created. It is ready to be filed.",
        "filed": "The document and its summary have been filed in the archive. The summary can now be sent.",
        "sent": "The summary has been sent to the relevant stakeholders. The document workflow is complete.",
    }

    @property
    def name(self) -> str:
        return "DocumentWorkflow"

    @property
    def goal(self) -> str:
        return "document summarized and filed"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "sent"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        # Distractor: state stays the same, step is wasted
        return StepResult(
            state=self._state + f" (You tried '{action}' but it had no useful effect.)",
            done=False,
        )


class CodeReview(TextEnvironment):
    """Review a set of code changes through a structured process.

    Optimal path (6 steps):
        read_diff -> run_tests -> identify_issues -> write_comments -> request_changes -> verify_fixes
    """

    STAGES = [
        "pending",
        "diff_read",
        "tests_run",
        "issues_identified",
        "comments_written",
        "changes_requested",
        "verified",
    ]

    OPTIMAL_PATH = [
        "read_diff",
        "run_tests",
        "identify_issues",
        "write_comments",
        "request_changes",
        "verify_fixes",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "pending": [
            "approve_immediately",
            "assign_to_other",
            "close_pr",
            "add_label",
            "check_ci_status",
        ],
        "diff_read": [
            "rebase_branch",
            "cherry_pick_commit",
            "view_blame",
            "check_coverage",
            "open_in_ide",
        ],
        "tests_run": [
            "add_more_tests",
            "skip_failing_tests",
            "regenerate_snapshots",
            "update_fixtures",
        ],
        "issues_identified": [
            "ignore_issues",
            "auto_fix_lint",
            "suggest_refactor",
            "create_follow_up_issue",
            "benchmark_performance",
        ],
        "comments_written": [
            "delete_comments",
            "resolve_all",
            "add_emoji_reactions",
            "tag_maintainer",
        ],
        "changes_requested": [
            "merge_anyway",
            "dismiss_review",
            "convert_to_draft",
            "squash_commits",
            "revert_changes",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "pending": "A pull request with 5 changed files is awaiting your review. You have not looked at the changes yet.",
        "diff_read": "You have read through all the diffs. Several files show significant logic changes.",
        "tests_run": "The test suite has been executed. 2 tests fail and coverage dropped by 3%.",
        "issues_identified": "You have identified 3 issues: a potential null dereference, a missing edge case, and a style violation.",
        "comments_written": "Inline comments have been posted on the relevant lines explaining each issue.",
        "changes_requested": "The review has been submitted with a 'request changes' status. The author is notified.",
        "verified": "The author has addressed all issues. Tests pass and coverage is restored. Review completed.",
    }

    @property
    def name(self) -> str:
        return "CodeReview"

    @property
    def goal(self) -> str:
        return "review completed with all issues addressed"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "verified"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the review.)",
            done=False,
        )


class EmailTriage(TextEnvironment):
    """Process a batch of incoming emails through triage.

    Optimal path (6 steps):
        scan_inbox -> read_urgent -> reply_urgent -> categorize_remaining
        -> forward_delegated -> archive_processed
    """

    STAGES = [
        "inbox_full",
        "scanned",
        "urgent_read",
        "urgent_replied",
        "categorized",
        "forwarded",
        "archived",
    ]

    OPTIMAL_PATH = [
        "scan_inbox",
        "read_urgent",
        "reply_urgent",
        "categorize_remaining",
        "forward_delegated",
        "archive_processed",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "inbox_full": [
            "delete_all",
            "mark_all_read",
            "sort_by_sender",
            "create_filter",
            "enable_vacation_reply",
        ],
        "scanned": [
            "read_newsletters",
            "unsubscribe_all",
            "flag_random",
            "export_to_csv",
        ],
        "urgent_read": [
            "snooze_email",
            "move_to_later",
            "print_email",
            "add_to_calendar",
            "translate_email",
        ],
        "urgent_replied": [
            "send_follow_up",
            "create_task_from_email",
            "cc_manager",
            "set_reminder",
        ],
        "categorized": [
            "re_categorize",
            "merge_threads",
            "apply_template",
            "generate_report",
            "search_similar",
        ],
        "forwarded": [
            "recall_forward",
            "add_attachment",
            "set_priority",
            "compose_new",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "inbox_full": "Your inbox contains 12 unread emails. Some appear urgent based on subject lines.",
        "scanned": "You have scanned all subject lines. 3 emails are marked urgent, 5 are routine, 4 are newsletters.",
        "urgent_read": "You have read the 3 urgent emails. One requires an immediate response about a deadline change.",
        "urgent_replied": "You have replied to the urgent emails. The deadline change has been acknowledged.",
        "categorized": "Remaining emails have been categorized: 2 need delegation, 3 are informational, 4 are newsletters.",
        "forwarded": "Delegated emails have been forwarded to the appropriate team members with context.",
        "archived": "All processed emails have been archived. Inbox is clean. All urgent items handled.",
    }

    @property
    def name(self) -> str:
        return "EmailTriage"

    @property
    def goal(self) -> str:
        return "all emails triaged and urgent ones replied to"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "archived"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it was not the right next step.)",
            done=False,
        )


class DataPipeline(TextEnvironment):
    """Run a data pipeline from validation to deployment.

    Optimal path (5 steps):
        validate_input -> transform_data -> run_quality_checks -> stage_output -> deploy_pipeline
    """

    STAGES = [
        "raw_input",
        "validated",
        "transformed",
        "checked",
        "staged",
        "deployed",
    ]

    OPTIMAL_PATH = [
        "validate_input",
        "transform_data",
        "run_quality_checks",
        "stage_output",
        "deploy_pipeline",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "raw_input": [
            "skip_validation",
            "sample_data",
            "profile_schema",
            "backup_raw",
            "compress_input",
        ],
        "validated": [
            "re_validate",
            "log_statistics",
            "normalize_encoding",
            "split_into_shards",
            "visualize_distribution",
        ],
        "transformed": [
            "rollback_transform",
            "export_intermediate",
            "benchmark_throughput",
            "add_lineage_tags",
        ],
        "checked": [
            "ignore_warnings",
            "generate_report",
            "compare_to_baseline",
            "send_alert",
            "retry_failed_checks",
        ],
        "staged": [
            "unstage",
            "dry_run",
            "notify_stakeholders",
            "update_documentation",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "raw_input": "A new data batch (50k rows) has arrived. Schema and data quality are unknown.",
        "validated": "Input validation passed. Schema matches expectations, 12 null values found and handled.",
        "transformed": "Data has been transformed: dates normalized, categories encoded, features engineered.",
        "checked": "Quality checks passed. No anomalies detected, distributions match historical baselines.",
        "staged": "Output has been staged in the pre-production environment. Ready for final deployment.",
        "deployed": "Pipeline deployed successfully. Data is live in production. Monitoring enabled.",
    }

    @property
    def name(self) -> str:
        return "DataPipeline"

    @property
    def goal(self) -> str:
        return "pipeline validated and deployed"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "deployed"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the pipeline.)",
            done=False,
        )


class ResearchTask(TextEnvironment):
    """Conduct a literature research task from search to synthesis.

    Optimal path (5 steps):
        search_papers -> read_abstracts -> take_notes -> identify_themes -> synthesize_summary
    """

    STAGES = [
        "topic_given",
        "papers_found",
        "abstracts_read",
        "notes_taken",
        "themes_identified",
        "synthesized",
    ]

    OPTIMAL_PATH = [
        "search_papers",
        "read_abstracts",
        "take_notes",
        "identify_themes",
        "synthesize_summary",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "topic_given": [
            "refine_topic",
            "check_wikipedia",
            "ask_colleague",
            "browse_twitter",
            "set_up_alerts",
        ],
        "papers_found": [
            "download_all_pdfs",
            "sort_by_citations",
            "check_retracted",
            "export_bibtex",
            "read_full_texts",
        ],
        "abstracts_read": [
            "re_read_abstracts",
            "highlight_keywords",
            "compare_methods",
            "build_citation_graph",
        ],
        "notes_taken": [
            "reorganize_notes",
            "color_code_themes",
            "create_timeline",
            "map_authors",
            "generate_word_cloud",
        ],
        "themes_identified": [
            "add_more_themes",
            "create_taxonomy",
            "draw_diagram",
            "write_outline",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "topic_given": "You need to research 'agentic AI planning methods'. No papers have been gathered yet.",
        "papers_found": "A search returned 25 relevant papers spanning 2020-2025. Titles and metadata are available.",
        "abstracts_read": "You have read all 25 abstracts. Key approaches include MCTS, LLM-based planning, and world models.",
        "notes_taken": "Detailed notes captured for each paper: methods, datasets, results, limitations.",
        "themes_identified": "Four main themes emerged: (1) search-based, (2) LLM reasoning, (3) world models, (4) hybrid approaches.",
        "synthesized": "A coherent summary synthesizes all themes with key findings and open questions. Research complete.",
    }

    @property
    def name(self) -> str:
        return "ResearchTask"

    @property
    def goal(self) -> str:
        return "research synthesized into summary"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "synthesized"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the research.)",
            done=False,
        )


# ======================================================================
# TEST-ONLY environments (never seen during training)
# ======================================================================


class CustomerSupport(TextEnvironment):
    """Handle a customer support ticket through resolution.

    Optimal path (6 steps):
        read_ticket -> check_account -> diagnose_issue -> apply_fix
        -> verify_resolution -> close_ticket
    """

    STAGES = [
        "new_ticket",
        "ticket_read",
        "account_checked",
        "diagnosed",
        "fixed",
        "verified",
        "closed",
    ]

    OPTIMAL_PATH = [
        "read_ticket",
        "check_account",
        "diagnose_issue",
        "apply_fix",
        "verify_resolution",
        "close_ticket",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "new_ticket": [
            "escalate_immediately",
            "send_template_response",
            "merge_with_other_ticket",
            "change_priority",
            "assign_to_team",
        ],
        "ticket_read": [
            "ask_for_more_info",
            "search_knowledge_base",
            "check_similar_tickets",
            "tag_ticket",
        ],
        "account_checked": [
            "reset_password",
            "check_billing",
            "review_past_tickets",
            "update_contact_info",
            "check_subscription",
        ],
        "diagnosed": [
            "document_workaround",
            "create_bug_report",
            "request_engineering_help",
            "rollback_account",
        ],
        "fixed": [
            "apply_additional_fix",
            "run_diagnostics",
            "update_knowledge_base",
            "schedule_follow_up",
            "offer_compensation",
        ],
        "verified": [
            "reopen_ticket",
            "send_survey",
            "add_internal_note",
            "flag_for_review",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "new_ticket": "A new support ticket has arrived. Subject: 'Cannot access my dashboard after update'. Priority: High.",
        "ticket_read": "You have read the ticket details. The customer reports a 403 error when accessing the dashboard since yesterday's update.",
        "account_checked": "Account status verified: active subscription, no payment issues. Last successful login was 2 days ago before the update.",
        "diagnosed": "Root cause identified: the update changed permission scopes and this customer's role was not migrated correctly.",
        "fixed": "Permission scopes have been updated for the customer's account. The role migration has been applied.",
        "verified": "Customer confirmed they can now access the dashboard. The fix is working correctly.",
        "closed": "Ticket closed with resolution notes. Customer satisfied. Support case complete.",
    }

    @property
    def name(self) -> str:
        return "CustomerSupport"

    @property
    def goal(self) -> str:
        return "customer issue resolved and ticket closed"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "closed"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not resolve the issue.)",
            done=False,
        )


class IncidentResponse(TextEnvironment):
    """Respond to a production incident through structured resolution.

    Optimal path (7 steps):
        acknowledge_alert -> assess_severity -> identify_affected_systems
        -> apply_mitigation -> root_cause_analysis -> implement_fix -> write_postmortem
    """

    STAGES = [
        "alert_fired",
        "acknowledged",
        "severity_assessed",
        "systems_identified",
        "mitigated",
        "root_cause_found",
        "fix_deployed",
        "postmortem_done",
    ]

    OPTIMAL_PATH = [
        "acknowledge_alert",
        "assess_severity",
        "identify_affected_systems",
        "apply_mitigation",
        "root_cause_analysis",
        "implement_fix",
        "write_postmortem",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "alert_fired": [
            "snooze_alert",
            "check_grafana",
            "ping_on_call",
            "open_slack_channel",
            "silence_pagerduty",
        ],
        "acknowledged": [
            "rollback_immediately",
            "check_recent_deploys",
            "notify_stakeholders",
            "start_status_page",
        ],
        "severity_assessed": [
            "upgrade_severity",
            "page_more_engineers",
            "check_customer_reports",
            "review_runbook",
            "spin_up_war_room",
        ],
        "systems_identified": [
            "isolate_network",
            "scale_up_replicas",
            "enable_debug_logging",
            "check_dependencies",
        ],
        "mitigated": [
            "verify_metrics",
            "update_status_page",
            "notify_customers",
            "collect_logs",
            "check_data_integrity",
        ],
        "root_cause_found": [
            "write_hotfix",
            "revert_commit",
            "patch_config",
            "add_monitoring",
        ],
        "fix_deployed": [
            "run_load_test",
            "verify_in_staging",
            "update_docs",
            "schedule_review",
            "close_incident",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "alert_fired": "PagerDuty alert: API latency p99 spiked to 12s (threshold: 2s). Error rate at 15%. 3 minutes ago.",
        "acknowledged": "Alert acknowledged. You are the incident commander. Timer started.",
        "severity_assessed": "Severity: SEV-1. Customer-facing API degraded. Estimated 30% of requests failing. Revenue impact confirmed.",
        "systems_identified": "Affected: payment-service, checkout-api, order-processor. Root in payment-service DB connection pool exhaustion.",
        "mitigated": "Immediate mitigation applied: traffic shifted to backup region, connection pool limits increased. Error rate dropping.",
        "root_cause_found": "Root cause: a config change in yesterday's deploy reduced max DB connections from 100 to 10. Change author identified.",
        "fix_deployed": "Config reverted and deployed. Connection pool restored. All metrics back to normal. Monitoring confirms stability.",
        "postmortem_done": "Postmortem written with timeline, root cause, impact analysis, and 4 action items. Incident fully resolved.",
    }

    @property
    def name(self) -> str:
        return "IncidentResponse"

    @property
    def goal(self) -> str:
        return "incident resolved with postmortem completed"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "postmortem_done"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the incident response.)",
            done=False,
        )


class MeetingPreparation(TextEnvironment):
    """Prepare for an important stakeholder meeting.

    Optimal path (5 steps):
        review_agenda -> gather_data -> prepare_slides -> rehearse_presentation
        -> send_pre_read
    """

    STAGES = [
        "meeting_scheduled",
        "agenda_reviewed",
        "data_gathered",
        "slides_ready",
        "rehearsed",
        "pre_read_sent",
    ]

    OPTIMAL_PATH = [
        "review_agenda",
        "gather_data",
        "prepare_slides",
        "rehearse_presentation",
        "send_pre_read",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "meeting_scheduled": [
            "reschedule_meeting",
            "invite_more_people",
            "book_different_room",
            "check_conflicts",
            "decline_meeting",
        ],
        "agenda_reviewed": [
            "rewrite_agenda",
            "add_agenda_items",
            "send_agenda_to_all",
            "create_sub_meetings",
        ],
        "data_gathered": [
            "request_more_data",
            "build_dashboard",
            "run_additional_analysis",
            "format_as_report",
            "cross_reference_sources",
        ],
        "slides_ready": [
            "add_animations",
            "change_template",
            "add_backup_slides",
            "export_to_pdf",
        ],
        "rehearsed": [
            "rehearse_again",
            "record_practice",
            "get_feedback",
            "revise_talking_points",
            "prepare_handouts",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "meeting_scheduled": "A quarterly business review meeting is scheduled for tomorrow at 10am. 8 stakeholders attending. No preparation done yet.",
        "agenda_reviewed": "Agenda reviewed: 4 topics — Q4 results, roadmap update, budget request, hiring plan. You need data for each.",
        "data_gathered": "Key metrics collected: revenue up 12%, 3 features shipped, 2 delayed. Budget utilization at 87%. Hiring: 4 open positions.",
        "slides_ready": "Presentation deck created with 15 slides covering all agenda items. Charts and key takeaways included.",
        "rehearsed": "Presentation rehearsed. Timing is 25 minutes with 5 minutes buffer for Q&A. Key objections anticipated.",
        "pre_read_sent": "Pre-read materials sent to all stakeholders. Meeting preparation is complete.",
    }

    @property
    def name(self) -> str:
        return "MeetingPreparation"

    @property
    def goal(self) -> str:
        return "meeting fully prepared with pre-read sent"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "pre_read_sent"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the preparation.)",
            done=False,
        )


# ======================================================================
# New environments with rich natural language action descriptions
# ======================================================================


class BugTriage(TextEnvironment):
    """Triage and fix a reported software bug.

    Optimal path (6 steps):
        reproduce_bug -> analyze_logs -> identify_root_cause -> write_fix
        -> run_regression_tests -> deploy_hotfix
    """

    STAGES = [
        "bug_reported",
        "reproduced",
        "logs_analyzed",
        "root_cause_found",
        "fix_written",
        "tests_passed",
        "hotfix_deployed",
    ]

    OPTIMAL_PATH = [
        "Reproduce the bug by following the steps described in the report",
        "Analyze the application logs to find error traces and stack dumps",
        "Identify the root cause by tracing the error back to the faulty code path",
        "Write a targeted code fix that addresses the root cause without side effects",
        "Run the full regression test suite to verify the fix does not break anything",
        "Deploy the hotfix to production through the standard release pipeline",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "bug_reported": [
            "Close the bug report as a duplicate without checking",
            "Reassign the ticket to another team member",
            "Lower the priority of the bug to non-critical",
            "Add a comment asking for more information",
            "Mark the bug as works-on-my-machine",
        ],
        "reproduced": [
            "Try to reproduce the bug on a different operating system",
            "Record a screencast of the reproduction steps",
            "Write a unit test for an unrelated feature",
            "Update the bug report with screenshots",
        ],
        "logs_analyzed": [
            "Enable verbose debug logging for all services",
            "Search for similar errors in the knowledge base",
            "Rotate the log files to free up disk space",
            "Forward the logs to the security team for review",
            "Archive old log entries to cold storage",
        ],
        "root_cause_found": [
            "Refactor the surrounding code for better readability",
            "Write a detailed technical design document about the issue",
            "Add extra logging around the problem area for future debugging",
            "Create a Jira epic to track the broader technical debt",
        ],
        "fix_written": [
            "Rewrite the entire module from scratch",
            "Add performance benchmarks for the changed code",
            "Request a code review from three separate reviewers",
            "Squash all commits into a single clean commit",
            "Update the changelog with the fix details",
        ],
        "tests_passed": [
            "Run a load test against the staging environment",
            "Wait for the next scheduled release window",
            "Send a notification to all stakeholders about the fix",
            "Create a rollback plan in case the fix fails",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "bug_reported": "A critical bug report has been filed: users cannot save their work. 47 users affected in the last hour. No reproduction steps confirmed yet.",
        "reproduced": "Bug reproduced consistently: clicking Save triggers a 500 error. The network tab shows the API returns an internal server error on POST /api/documents/save.",
        "logs_analyzed": "Log analysis reveals a NullPointerException in DocumentService.save() at line 234. The error started after yesterday's deployment of version 2.14.0.",
        "root_cause_found": "Root cause identified: a database migration in v2.14.0 added a NOT NULL constraint on the 'updated_by' column, but the save endpoint does not populate this field for anonymous users.",
        "fix_written": "Fix implemented: the save endpoint now populates 'updated_by' with a default system user for anonymous sessions. The migration has been updated to allow NULL as a fallback.",
        "tests_passed": "All 342 regression tests pass. The specific save scenario for anonymous users now works correctly. No existing tests broken.",
        "hotfix_deployed": "Hotfix v2.14.1 deployed to production. Save functionality restored for all users. Error rate dropped to 0%. Bug resolved.",
    }

    @property
    def name(self) -> str:
        return "BugTriage"

    @property
    def goal(self) -> str:
        return "bug fixed and hotfix deployed to production"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "hotfix_deployed"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not fix the bug.)",
            done=False,
        )


class OnboardingProcess(TextEnvironment):
    """Onboard a new employee through HR processes.

    Optimal path (5 steps):
        create_account -> assign_equipment -> schedule_orientation
        -> set_up_workspace -> complete_checklist
    """

    STAGES = [
        "hire_confirmed",
        "account_created",
        "equipment_assigned",
        "orientation_scheduled",
        "workspace_ready",
        "onboarding_complete",
    ]

    OPTIMAL_PATH = [
        "Create the employee's corporate accounts including email and Slack",
        "Assign a laptop and necessary peripherals from the IT inventory",
        "Schedule the two-day orientation program with HR and the team lead",
        "Set up the physical workspace with desk, chair, and access badge",
        "Complete and sign off the onboarding checklist with the manager",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "hire_confirmed": [
            "Send a welcome email before accounts are ready",
            "Post about the new hire on the company blog",
            "Order business cards for the new employee",
            "Schedule a team lunch for next month",
            "Update the organizational chart",
        ],
        "account_created": [
            "Grant admin access to all internal systems",
            "Enroll the employee in all optional training courses",
            "Set up a personal website for the employee",
            "Create shared folders for every department",
        ],
        "equipment_assigned": [
            "Order additional monitors as backup",
            "Install non-standard software on the laptop",
            "Ship equipment to a secondary office location",
            "Purchase premium noise-canceling headphones",
            "Set up a personal VPN profile",
        ],
        "orientation_scheduled": [
            "Reschedule orientation to next quarter",
            "Invite all company executives to the orientation",
            "Create a custom orientation video",
            "Prepare a 50-page onboarding manual",
        ],
        "workspace_ready": [
            "Redesign the entire office floor plan",
            "Order custom furniture for the workspace",
            "Install additional security cameras near the desk",
            "Rearrange the team seating arrangement",
            "Set up a dedicated break room",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "hire_confirmed": "New hire confirmed: Maria Garcia, Software Engineer, starting next Monday. No accounts or equipment have been prepared yet.",
        "account_created": "Corporate accounts created: maria.garcia@company.com, Slack, GitHub, and Jira access provisioned. MFA enrolled. Waiting for equipment.",
        "equipment_assigned": "MacBook Pro and 27-inch monitor assigned from IT inventory. Serial numbers recorded. Software image ready to deploy on first day.",
        "orientation_scheduled": "Two-day orientation scheduled for Monday-Tuesday: Day 1 company overview with HR, Day 2 team introduction with engineering lead.",
        "workspace_ready": "Desk B-204 set up with ergonomic chair, dual monitors, and docking station. Building access badge #4521 programmed and ready.",
        "onboarding_complete": "Onboarding checklist signed off by both the new hire and manager. Maria is fully set up and ready for her first day. Process complete.",
    }

    @property
    def name(self) -> str:
        return "OnboardingProcess"

    @property
    def goal(self) -> str:
        return "new employee fully onboarded and ready to start"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "onboarding_complete"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the onboarding.)",
            done=False,
        )


class SecurityAudit(TextEnvironment):
    """Conduct a security audit of a web application.

    Optimal path (6 steps):
        scope_assessment -> vulnerability_scan -> manual_testing
        -> risk_classification -> remediation_plan -> executive_report
    """

    STAGES = [
        "audit_requested",
        "scope_defined",
        "scan_complete",
        "manual_done",
        "risks_classified",
        "remediation_planned",
        "report_delivered",
    ]

    OPTIMAL_PATH = [
        "Define the scope of the assessment including target systems and testing boundaries",
        "Run automated vulnerability scanners against all in-scope web endpoints",
        "Perform manual penetration testing on critical authentication and authorization flows",
        "Classify discovered vulnerabilities by risk level using CVSS scoring methodology",
        "Develop a prioritized remediation plan with timelines for each finding",
        "Write and deliver the executive summary report with findings and recommendations",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "audit_requested": [
            "Start testing immediately without defining scope",
            "Request budget approval for expensive third-party tools",
            "Schedule the audit for next quarter instead",
            "Review last year's audit report for reference",
            "Notify all development teams about the upcoming audit",
        ],
        "scope_defined": [
            "Expand scope to include all company infrastructure",
            "Purchase additional security licenses",
            "Set up a dedicated testing network",
            "Create detailed documentation of all API endpoints",
        ],
        "scan_complete": [
            "Re-run the scanner with more aggressive settings",
            "Share raw scan results with the development team",
            "Compare scan results with industry benchmarks",
            "Set up continuous monitoring for new vulnerabilities",
            "Archive scan results in the compliance database",
        ],
        "manual_done": [
            "Attempt to exploit every finding for proof of concept",
            "Document each testing step in exhaustive detail",
            "Research zero-day vulnerabilities in the tech stack",
            "Set up a bug bounty program for external testers",
        ],
        "risks_classified": [
            "Reclassify all findings as critical to get faster action",
            "Cross-reference findings with threat intelligence feeds",
            "Calculate the potential financial impact of each risk",
            "Create a risk dashboard for continuous monitoring",
            "Present preliminary findings to the board of directors",
        ],
        "remediation_planned": [
            "Begin implementing fixes before report approval",
            "Hire additional consultants for remediation support",
            "Schedule a follow-up audit for next month",
            "Create training materials based on the findings",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "audit_requested": "Security audit requested for the customer-facing web application. No scope or testing plan has been defined yet.",
        "scope_defined": "Audit scope defined: 3 web applications, 12 API endpoints, authentication system, and payment processing flow. Testing window: 2 weeks.",
        "scan_complete": "Automated scan complete: 156 findings detected. 8 critical, 23 high, 45 medium, 80 low severity. Manual validation needed.",
        "manual_done": "Manual testing complete: confirmed 6 of 8 critical findings. Discovered 2 additional IDOR vulnerabilities and a session fixation bug not caught by scanners.",
        "risks_classified": "All 160 findings classified: 8 critical (CVSS 9.0+), 25 high (7.0-8.9), 47 medium (4.0-6.9), 80 low (<4.0). Critical findings affect payment flow.",
        "remediation_planned": "Remediation plan complete: critical fixes within 48 hours, high within 2 weeks, medium within 30 days. Development team assigned to each finding.",
        "report_delivered": "Executive report delivered to CISO and CTO. Contains findings summary, risk analysis, remediation timeline, and compliance impact. Audit complete.",
    }

    @property
    def name(self) -> str:
        return "SecurityAudit"

    @property
    def goal(self) -> str:
        return "security audit completed with report delivered"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "report_delivered"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the audit.)",
            done=False,
        )


class ContentPublishing(TextEnvironment):
    """Publish a blog post through editorial workflow.

    Optimal path (5 steps):
        draft_article -> editorial_review -> add_media
        -> seo_optimization -> publish_and_promote
    """

    STAGES = [
        "topic_assigned",
        "draft_ready",
        "review_passed",
        "media_added",
        "seo_optimized",
        "published",
    ]

    OPTIMAL_PATH = [
        "Write the full article draft based on the assigned topic and target audience",
        "Submit the draft for editorial review and incorporate all feedback",
        "Add relevant images, diagrams, and alt text to support the article content",
        "Optimize the article for search engines with keywords, meta tags, and internal links",
        "Publish the article and share it across social media and newsletter channels",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "topic_assigned": [
            "Research competitors' articles on the same topic",
            "Create a detailed content calendar for the next quarter",
            "Brainstorm ten alternative angles for the article",
            "Set up analytics tracking before writing anything",
            "Interview subject matter experts for background quotes",
        ],
        "draft_ready": [
            "Rewrite the entire article in a different tone",
            "Add footnotes and academic citations throughout",
            "Translate the article into three additional languages",
            "Create a companion podcast episode",
        ],
        "review_passed": [
            "Request a second round of review from a different editor",
            "Run the article through an AI writing detector",
            "Create an infographic summarizing the key points",
            "Record a video summary of the article",
            "Fact-check every claim against primary sources",
        ],
        "media_added": [
            "Commission custom illustrations from a designer",
            "Create an interactive data visualization",
            "Add animated GIFs to every section",
            "Produce a short documentary about the topic",
        ],
        "seo_optimized": [
            "Purchase backlinks from external websites",
            "Rewrite the article to be exactly 2000 words",
            "Create a separate landing page for the article",
            "Set up A/B testing for the headline",
            "Build a topic cluster with five supporting articles",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "topic_assigned": "New article assignment: 'How AI is Transforming Supply Chain Management'. Target: 1500 words, business audience. Deadline: Friday.",
        "draft_ready": "Article draft complete at 1650 words. Covers three key use cases with industry examples. Needs editorial review.",
        "review_passed": "Editorial review passed with minor revisions. Grammar corrected, flow improved, one section restructured. Ready for media.",
        "media_added": "Three relevant images added with descriptive alt text. One process diagram created. Thumbnail designed for social sharing.",
        "seo_optimized": "SEO optimized: primary keyword density at 1.5%, meta description written, 4 internal links added, URL slug finalized.",
        "published": "Article published and promoted. Shared on LinkedIn, Twitter, and company newsletter. Analytics tracking confirmed. Content workflow complete.",
    }

    @property
    def name(self) -> str:
        return "ContentPublishing"

    @property
    def goal(self) -> str:
        return "article published and promoted across channels"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "published"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the publishing process.)",
            done=False,
        )


class ExperimentPipeline(TextEnvironment):
    """Run a machine learning experiment from hypothesis to analysis.

    Optimal path (6 steps):
        formulate_hypothesis -> prepare_dataset -> train_model
        -> evaluate_results -> statistical_testing -> write_findings
    """

    STAGES = [
        "research_question",
        "hypothesis_ready",
        "dataset_prepared",
        "model_trained",
        "results_evaluated",
        "significance_tested",
        "findings_written",
    ]

    OPTIMAL_PATH = [
        "Formulate a testable hypothesis with clear dependent and independent variables",
        "Prepare and preprocess the dataset with proper train-validation-test splits",
        "Train the model with the specified architecture and hyperparameters",
        "Evaluate the trained model on the held-out test set using predefined metrics",
        "Run statistical significance tests to validate the observed improvements",
        "Write up the findings with tables, figures, and interpretation of results",
    ]

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "research_question": [
            "Survey all related work before forming any hypothesis",
            "Design the experiment for maximum publication impact",
            "Choose the most complex model architecture available",
            "Set up distributed training infrastructure first",
            "Define twenty different metrics to track",
        ],
        "hypothesis_ready": [
            "Revise the hypothesis to be less falsifiable",
            "Add three additional hypotheses to test simultaneously",
            "Review competing hypotheses from other research groups",
            "Write the introduction section of the paper first",
        ],
        "dataset_prepared": [
            "Augment the dataset with synthetic examples",
            "Collect additional data from new sources",
            "Visualize the entire dataset distribution",
            "Create a custom data loader with advanced caching",
            "Re-annotate ambiguous examples in the dataset",
        ],
        "model_trained": [
            "Tune hyperparameters exhaustively with grid search",
            "Train an ensemble of ten model variants",
            "Profile the model's memory usage and inference speed",
            "Distill the model into a smaller architecture",
        ],
        "results_evaluated": [
            "Cherry-pick the best-performing checkpoint",
            "Run the evaluation on a different random seed",
            "Visualize attention patterns and model internals",
            "Compare against twenty additional baselines",
            "Compute per-class breakdown for all metrics",
        ],
        "significance_tested": [
            "Apply Bayesian analysis in addition to frequentist tests",
            "Compute effect sizes and confidence intervals",
            "Run a meta-analysis across related studies",
            "Create publication-quality plots for every metric",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "research_question": "Research question defined: Does adding contrastive pre-training improve downstream classification on low-resource datasets? No hypothesis formulated yet.",
        "hypothesis_ready": "Hypothesis: Contrastive pre-training will improve F1 by at least 3 points on datasets with fewer than 1000 labeled examples, compared to the standard fine-tuning baseline.",
        "dataset_prepared": "Dataset prepared: 3 low-resource classification benchmarks (800, 500, and 200 labeled examples). 80/10/10 train/val/test splits. Tokenization and feature extraction complete.",
        "model_trained": "Model trained for 50 epochs. Training converged at epoch 38. Best validation F1: 0.72 (contrastive) vs 0.67 (baseline). Training logs and checkpoints saved.",
        "results_evaluated": "Test set evaluation complete. Contrastive model: F1=0.71, Accuracy=0.74. Baseline: F1=0.66, Accuracy=0.69. Improvement of 5 F1 points across all three benchmarks.",
        "significance_tested": "Paired t-test across 5 seeds: p=0.003 (significant at alpha=0.01). Effect size (Cohen's d) = 1.2 (large). Bootstrap confidence interval for F1 difference: [3.1, 6.8] points.",
        "findings_written": "Findings written up: contrastive pre-training yields statistically significant improvements on low-resource classification. Tables, learning curves, and ablation results included. Experiment complete.",
    }

    @property
    def name(self) -> str:
        return "ExperimentPipeline"

    @property
    def goal(self) -> str:
        return "experiment completed with findings written up"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "findings_written"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the experiment.)",
            done=False,
        )


# ======================================================================
# Non-linear environments (branching paths)
# ======================================================================


class ProjectPlanning(TextEnvironment):
    """Plan a software project with multiple valid paths.

    This environment has BRANCHING: after gathering requirements, the agent
    can choose either a technical design path OR a prototype-first path.
    Both paths converge at the implementation stage.

    Path A (design-first, 6 steps):
        gather_requirements -> write_tech_spec -> design_architecture
        -> implement_features -> run_qa -> release
    Path B (prototype-first, 6 steps):
        gather_requirements -> build_prototype -> collect_feedback
        -> implement_features -> run_qa -> release
    """

    # State graph (not a linear chain)
    GRAPH: dict[str, dict[str, str]] = {
        "start": {
            "Gather requirements from stakeholders through interviews and surveys": "requirements_done",
        },
        "requirements_done": {
            "Write a detailed technical specification document with system diagrams": "tech_spec_done",
            "Build a quick interactive prototype to validate the core user flow": "prototype_done",
        },
        "tech_spec_done": {
            "Design the system architecture with component diagrams and API contracts": "architecture_done",
        },
        "prototype_done": {
            "Collect user feedback on the prototype and document required changes": "feedback_done",
        },
        "architecture_done": {
            "Implement the core features following the technical specification": "implemented",
        },
        "feedback_done": {
            "Implement the core features incorporating the user feedback": "implemented",
        },
        "implemented": {
            "Run the full quality assurance test suite and fix critical bugs": "qa_done",
        },
        "qa_done": {
            "Release the software to production with monitoring enabled": "released",
        },
    }

    DISTRACTORS: dict[str, list[str]] = {
        "start": [
            "Start coding immediately without understanding the requirements",
            "Schedule a team offsite to brainstorm ideas",
            "Research competing products for inspiration",
            "Set up the development environment and CI pipeline",
            "Create a detailed project timeline with Gantt charts",
        ],
        "requirements_done": [
            "Rewrite all requirements in a different format",
            "Hire additional team members before proceeding",
            "Conduct a market analysis of competing solutions",
        ],
        "tech_spec_done": [
            "Rewrite the specification using a different template",
            "Get legal review of all technical decisions",
            "Benchmark three different database solutions",
            "Create a detailed cost analysis for cloud infrastructure",
        ],
        "prototype_done": [
            "Polish the prototype with pixel-perfect design",
            "Present the prototype at a company all-hands meeting",
            "Rewrite the prototype in a different programming language",
            "Add analytics tracking to the prototype",
        ],
        "architecture_done": [
            "Refactor the architecture to use microservices",
            "Write comprehensive documentation for every component",
            "Set up monitoring dashboards before writing any code",
            "Conduct a security review of the architecture",
        ],
        "feedback_done": [
            "Conduct another round of user interviews",
            "Redesign the entire user interface from scratch",
            "A/B test two different feedback implementation strategies",
            "Create a detailed feature prioritization matrix",
        ],
        "implemented": [
            "Add performance optimizations before testing",
            "Write a user manual for the new features",
            "Refactor the codebase for better maintainability",
            "Set up automated deployment pipelines",
        ],
        "qa_done": [
            "Run an additional round of load testing",
            "Get executive sign-off before release",
            "Create a rollback plan in case of issues",
            "Write release notes for the marketing team",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "start": "New project assigned: build a customer feedback dashboard. No planning has been done yet. Team of 4 engineers available.",
        "requirements_done": "Requirements gathered: 12 user stories identified, 3 key personas defined, success metrics established. Two paths forward: design-first or prototype-first.",
        "tech_spec_done": "Technical specification complete: REST API with 8 endpoints, React frontend, PostgreSQL database. Component dependencies mapped. Ready for architecture design.",
        "prototype_done": "Interactive prototype built: clickable mockup covering the 3 main user flows. Ready for stakeholder feedback before full implementation.",
        "architecture_done": "System architecture designed: 3-tier architecture with caching layer, event-driven notifications, and role-based access control. Ready for implementation.",
        "feedback_done": "Feedback collected: 15 stakeholders tested the prototype. Key changes: simplify the main dashboard, add export functionality, improve mobile layout. Ready for implementation.",
        "implemented": "Core features implemented: all user stories completed, 89% test coverage, code reviewed. Ready for QA testing.",
        "qa_done": "QA complete: 156 test cases passed, 3 critical bugs fixed, performance benchmarks met. Ready for production release.",
        "released": "Software released to production. Monitoring shows stable performance. 94% of target metrics met on day one. Project complete.",
    }

    # Oracle prefers path A (design-first) as the "optimal" path
    ORACLE_PATH: dict[str, str] = {
        "start": "Gather requirements from stakeholders through interviews and surveys",
        "requirements_done": "Write a detailed technical specification document with system diagrams",
        "tech_spec_done": "Design the system architecture with component diagrams and API contracts",
        "architecture_done": "Implement the core features following the technical specification",
        "implemented": "Run the full quality assurance test suite and fix critical bugs",
        "qa_done": "Release the software to production with monitoring enabled",
    }

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._current_node = "start"

    @property
    def name(self) -> str:
        return "ProjectPlanning"

    @property
    def goal(self) -> str:
        return "software project released to production"

    @property
    def oracle_steps(self) -> int:
        return 6

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._current_node = "start"
        self._state = self.STATE_DESCRIPTIONS[self._current_node]
        return self._state

    def get_valid_actions(self) -> list[str]:
        valid_transitions = list(self.GRAPH.get(self._current_node, {}).keys())
        distractors = self.DISTRACTORS.get(self._current_node, [])
        actions = valid_transitions + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.ORACLE_PATH.get(self._current_node)

    def _execute(self, action: str) -> StepResult:
        transitions = self.GRAPH.get(self._current_node, {})
        if action in transitions:
            self._current_node = transitions[action]
            done = self._current_node == "released"
            return StepResult(state=self.STATE_DESCRIPTIONS[self._current_node], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the project.)",
            done=False,
        )


class TroubleshootingGuide(TextEnvironment):
    """Diagnose and fix a system issue with branching diagnostic paths.

    The agent must choose between checking hardware OR software first.
    Each path leads to different intermediate states but converges at the fix.

    Path A (hardware-first, 5 steps):
        check_hardware -> inspect_connections -> fix_hardware_issue
        -> verify_system -> document_resolution
    Path B (software-first, 5 steps):
        check_software -> analyze_logs -> fix_software_issue
        -> verify_system -> document_resolution
    """

    GRAPH: dict[str, dict[str, str]] = {
        "start": {
            "Check the hardware components for visible damage or loose connections": "hardware_checked",
            "Check the software logs and system configuration for errors": "software_checked",
        },
        "hardware_checked": {
            "Inspect all cable connections and reseat components that appear loose": "connections_inspected",
        },
        "software_checked": {
            "Analyze the error logs to identify the root cause of the failure": "logs_analyzed",
        },
        "connections_inspected": {
            "Replace the faulty hardware component and test the connection": "hardware_fixed",
        },
        "logs_analyzed": {
            "Apply the software patch that addresses the identified error": "software_fixed",
        },
        "hardware_fixed": {
            "Run a full system verification test to confirm the fix works": "verified",
        },
        "software_fixed": {
            "Run a full system verification test to confirm the fix works": "verified",
        },
        "verified": {
            "Document the resolution steps and update the knowledge base": "documented",
        },
    }

    DISTRACTORS: dict[str, list[str]] = {
        "start": [
            "Restart the entire system and hope the problem goes away",
            "Escalate the issue to a senior technician immediately",
            "Order replacement parts before diagnosing the problem",
            "Search the internet for similar issues reported by other users",
        ],
        "hardware_checked": [
            "Run a full diagnostic scan on all hardware components",
            "Replace all cables preventatively",
            "Check the power supply voltage with a multimeter",
            "Photograph the hardware configuration for documentation",
        ],
        "software_checked": [
            "Reinstall the operating system from scratch",
            "Roll back to the previous software version",
            "Run a virus scan on the entire system",
            "Clear all temporary files and caches",
        ],
        "connections_inspected": [
            "Order additional spare parts for future issues",
            "Run a stress test on the hardware components",
            "Upgrade the firmware on all connected devices",
            "Document the current hardware configuration",
        ],
        "logs_analyzed": [
            "Collect additional logs from related services",
            "Set up real-time log monitoring for future issues",
            "Compare logs with baseline performance metrics",
            "Forward the logs to the development team for review",
        ],
        "hardware_fixed": [
            "Run additional stress tests for 24 hours",
            "Replace preventatively other aging components",
            "Update the asset management database",
            "Schedule a follow-up inspection for next week",
        ],
        "software_fixed": [
            "Run additional regression tests on related modules",
            "Set up automated alerts for similar errors",
            "Review the change management process",
            "Create a backup of the current configuration",
        ],
        "verified": [
            "Run the verification test a second time for confidence",
            "Benchmark system performance against historical data",
            "Notify all affected users about the resolution",
            "Schedule preventive maintenance for next month",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "start": "System failure reported: the production server is unresponsive. Users cannot access the application. No diagnosis has been performed yet.",
        "hardware_checked": "Hardware inspection complete: no visible physical damage. One network cable appears slightly loose at the switch port. CPU and memory indicators are normal.",
        "software_checked": "Software check complete: system logs show repeated connection timeout errors starting 3 hours ago. Configuration files appear intact but a recent update may have introduced a bug.",
        "connections_inspected": "All connections inspected: the loose network cable was the primary suspect. The cable was reseated but the RJ45 connector shows signs of wear and may need replacement.",
        "logs_analyzed": "Log analysis complete: the timeout errors correlate with a configuration change made during yesterday's maintenance window. The DNS resolver setting was incorrectly modified.",
        "hardware_fixed": "Hardware fix applied: the worn network cable has been replaced with a new Cat6 cable. Initial connectivity test shows the link is stable at 1Gbps.",
        "software_fixed": "Software fix applied: the DNS resolver configuration has been corrected to point to the proper internal DNS servers. Service restart completed successfully.",
        "verified": "System verification complete: all services are responding normally. Application is accessible to users. Latency and throughput metrics are within expected ranges.",
        "documented": "Resolution documented: root cause, diagnostic steps, and fix have been recorded in the knowledge base. Ticket closed. Troubleshooting complete.",
    }

    ORACLE_PATH: dict[str, str] = {
        "start": "Check the hardware components for visible damage or loose connections",
        "hardware_checked": "Inspect all cable connections and reseat components that appear loose",
        "connections_inspected": "Replace the faulty hardware component and test the connection",
        "hardware_fixed": "Run a full system verification test to confirm the fix works",
        "verified": "Document the resolution steps and update the knowledge base",
    }

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._current_node = "start"

    @property
    def name(self) -> str:
        return "TroubleshootingGuide"

    @property
    def goal(self) -> str:
        return "system issue resolved and documented"

    @property
    def oracle_steps(self) -> int:
        return 5

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._current_node = "start"
        self._state = self.STATE_DESCRIPTIONS[self._current_node]
        return self._state

    def get_valid_actions(self) -> list[str]:
        valid_transitions = list(self.GRAPH.get(self._current_node, {}).keys())
        distractors = self.DISTRACTORS.get(self._current_node, [])
        actions = valid_transitions + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.ORACLE_PATH.get(self._current_node)

    def _execute(self, action: str) -> StepResult:
        transitions = self.GRAPH.get(self._current_node, {})
        if action in transitions:
            self._current_node = transitions[action]
            done = self._current_node == "documented"
            return StepResult(state=self.STATE_DESCRIPTIONS[self._current_node], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not help diagnose the issue.)",
            done=False,
        )


# ======================================================================
# Stochastic environments
# ======================================================================


class NoisyDeployment(TextEnvironment):
    """Deploy a service with stochastic failures.

    Some actions have a probability of failing and returning to a previous
    state. The agent must retry or adapt. This tests robustness to
    non-deterministic transitions.

    Optimal path (5 steps if no failures):
        build_artifact -> run_integration_tests -> deploy_to_staging
        -> run_smoke_tests -> promote_to_production
    """

    STAGES = [
        "code_ready",
        "artifact_built",
        "tests_passed",
        "staged",
        "smoke_passed",
        "in_production",
    ]

    OPTIMAL_PATH = [
        "Build the deployment artifact from the latest release branch",
        "Run the full integration test suite against the built artifact",
        "Deploy the artifact to the staging environment for validation",
        "Run smoke tests against the staging deployment to verify health",
        "Promote the staging deployment to production with blue-green switch",
    ]

    # Probability that each step FAILS and reverts to previous state
    FAILURE_PROBS: dict[int, float] = {
        0: 0.0,    # build always succeeds
        1: 0.1,    # integration tests: 10% flaky failure
        2: 0.15,   # deploy to staging: 15% failure (infra issues)
        3: 0.2,    # smoke tests: 20% failure (environment variance)
        4: 0.1,    # promote: 10% failure (health check timeout)
    }

    FAILURE_MESSAGES: dict[int, str] = {
        1: "Integration tests failed due to a flaky test. The test infrastructure had a transient network issue. You need to retry.",
        2: "Staging deployment failed: the container orchestrator rejected the pod due to resource limits. You need to retry.",
        3: "Smoke tests failed: the health check endpoint returned a 503 because the service was still initializing. You need to retry.",
        4: "Production promotion failed: the load balancer health check timed out during the blue-green switch. You need to retry.",
    }

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "code_ready": [
            "Run the linter one more time before building",
            "Update the version number in the changelog",
            "Cherry-pick an additional commit into the release branch",
            "Notify the team that a deployment is starting",
        ],
        "artifact_built": [
            "Scan the artifact for known security vulnerabilities",
            "Upload the artifact to a secondary registry as backup",
            "Verify the artifact checksum matches the source hash",
            "Tag the git commit with the artifact version number",
        ],
        "tests_passed": [
            "Run performance benchmarks before deploying",
            "Generate a test coverage report for the release",
            "Compare test results with the previous release",
            "Archive the test logs for compliance purposes",
        ],
        "staged": [
            "Load test the staging environment with synthetic traffic",
            "Compare staging metrics with production baseline",
            "Send a preview link to the product team for review",
            "Check that all feature flags are configured correctly",
        ],
        "smoke_passed": [
            "Wait for the next maintenance window to promote",
            "Run additional canary tests with a small traffic percentage",
            "Get explicit approval from the release manager",
            "Update the deployment documentation with current versions",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "code_ready": "Release branch v3.2.1 is ready with 12 commits. All CI checks passed. Ready to build the deployment artifact.",
        "artifact_built": "Docker image built and pushed to registry: app:v3.2.1-rc1 (245MB). Build took 4 minutes. Ready for testing.",
        "tests_passed": "All 1,247 integration tests passed in 12 minutes. Coverage: 91%. No regressions detected. Ready to deploy to staging.",
        "staged": "Deployed to staging environment (staging.internal). 3 replicas running, health checks passing. Ready for smoke tests.",
        "smoke_passed": "All 28 smoke tests passed on staging. API response times within SLA (p99 < 200ms). Ready for production promotion.",
        "in_production": "Successfully promoted to production via blue-green deployment. All health checks passing. Zero-downtime deployment complete.",
    }

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._stage_idx = 0

    @property
    def name(self) -> str:
        return "NoisyDeployment"

    @property
    def goal(self) -> str:
        return "service deployed to production successfully"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)  # Best case, no failures

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            # Check for stochastic failure
            fail_prob = self.FAILURE_PROBS.get(self._stage_idx, 0.0)
            # Use step_count to make failures reproducible but varied
            roll = random.Random(self._seed * 1000 + self._step_count).random()
            if roll < fail_prob:
                # Failure: stay at current stage, different state text
                fail_msg = self.FAILURE_MESSAGES.get(self._stage_idx, "Action failed. Retry needed.")
                return StepResult(state=fail_msg, done=False)

            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "in_production"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it did not advance the deployment.)",
            done=False,
        )


class UncertainDiagnosis(TextEnvironment):
    """Diagnose a patient with probabilistic test results.

    Test results are stochastic: a test may return inconclusive results,
    requiring the agent to either retry or choose a different diagnostic path.

    Optimal path (5 steps if no inconclusives):
        take_history -> order_blood_work -> request_imaging
        -> consult_specialist -> prescribe_treatment
    """

    STAGES = [
        "patient_arrived",
        "history_taken",
        "blood_results",
        "imaging_done",
        "specialist_consulted",
        "treatment_prescribed",
    ]

    OPTIMAL_PATH = [
        "Take a detailed patient history including symptoms and medical background",
        "Order comprehensive blood work including CBC and metabolic panel",
        "Request diagnostic imaging to visualize the affected area",
        "Consult with a specialist to interpret the combined findings",
        "Prescribe the appropriate treatment plan based on the diagnosis",
    ]

    # Probability of getting inconclusive results
    INCONCLUSIVE_PROBS: dict[int, float] = {
        0: 0.0,     # history always works
        1: 0.2,     # blood work: 20% inconclusive
        2: 0.15,    # imaging: 15% inconclusive
        3: 0.1,     # specialist: 10% needs more data
        4: 0.0,     # prescription always works
    }

    INCONCLUSIVE_MESSAGES: dict[int, str] = {
        1: "Blood work results are inconclusive: some values are borderline and need to be repeated. The lab recommends retesting in the same visit.",
        2: "Imaging results are unclear: the initial scan did not provide sufficient contrast. A repeat scan with adjusted parameters is recommended.",
        3: "The specialist requests additional information before making a recommendation. The current data is insufficient for a confident diagnosis.",
    }

    DISTRACTORS_BY_STAGE: dict[str, list[str]] = {
        "patient_arrived": [
            "Prescribe medication based on the initial complaint alone",
            "Refer the patient to a different department",
            "Schedule a follow-up appointment for next week",
            "Check the patient's insurance coverage first",
        ],
        "history_taken": [
            "Order a full-body MRI scan as a precaution",
            "Prescribe a general antibiotic while waiting for tests",
            "Consult the patient's previous records from other hospitals",
            "Administer a physical examination of unrelated systems",
        ],
        "blood_results": [
            "Order genetic testing for hereditary conditions",
            "Refer to a nutritionist for dietary assessment",
            "Prescribe supplements based on the blood results alone",
            "Request a second opinion on the blood work interpretation",
        ],
        "imaging_done": [
            "Order additional advanced imaging sequences",
            "Begin treatment based on imaging alone without specialist input",
            "Request a pathology review of incidental findings",
            "Schedule the patient for a biopsy procedure",
        ],
        "specialist_consulted": [
            "Seek a second specialist opinion from another institution",
            "Enroll the patient in a clinical trial",
            "Recommend watchful waiting instead of active treatment",
            "Order additional specialized testing",
        ],
    }

    STATE_DESCRIPTIONS: dict[str, str] = {
        "patient_arrived": "Patient presents with persistent chest discomfort and shortness of breath for 2 weeks. No prior cardiac history. Vitals: BP 145/90, HR 88, SpO2 96%.",
        "history_taken": "History complete: 52-year-old, non-smoker, sedentary lifestyle, family history of heart disease. Symptoms worsen with exertion. No fever or cough. Need diagnostic workup.",
        "blood_results": "Blood work results: elevated troponin (0.08 ng/mL), slightly elevated CRP, normal CBC. Lipid panel shows LDL 165 mg/dL. Findings suggest cardiac involvement.",
        "imaging_done": "Chest X-ray and echocardiogram complete: mild left ventricular hypertrophy, no effusion, ejection fraction 52% (low-normal). Coronary artery disease suspected.",
        "specialist_consulted": "Cardiology consultation complete: based on troponin elevation, imaging findings, and risk factors, diagnosis is stable angina with early coronary artery disease.",
        "treatment_prescribed": "Treatment plan prescribed: antiplatelet therapy, statin, beta-blocker, lifestyle modifications. Follow-up stress test in 6 weeks. Diagnosis and treatment complete.",
    }

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self._stage_idx = 0

    @property
    def name(self) -> str:
        return "UncertainDiagnosis"

    @property
    def goal(self) -> str:
        return "patient diagnosed and treatment prescribed"

    @property
    def oracle_steps(self) -> int:
        return len(self.OPTIMAL_PATH)

    def reset(self) -> str:
        self._rng = random.Random(self._seed)
        self._step_count = 0
        self._done = False
        self._stage_idx = 0
        self._state = self.STATE_DESCRIPTIONS[self.STAGES[self._stage_idx]]
        return self._state

    def get_valid_actions(self) -> list[str]:
        stage = self.STAGES[self._stage_idx]
        optimal = [self.OPTIMAL_PATH[self._stage_idx]]
        distractors = self.DISTRACTORS_BY_STAGE[stage]
        actions = optimal + distractors
        self._rng = random.Random(self._seed + self._step_count)
        self._rng.shuffle(actions)
        return actions

    def get_oracle_action(self) -> str | None:
        if self._done:
            return None
        return self.OPTIMAL_PATH[self._stage_idx]

    def _execute(self, action: str) -> StepResult:
        optimal = self.OPTIMAL_PATH[self._stage_idx]
        if action == optimal:
            # Check for inconclusive results
            inc_prob = self.INCONCLUSIVE_PROBS.get(self._stage_idx, 0.0)
            roll = random.Random(self._seed * 1000 + self._step_count).random()
            if roll < inc_prob:
                msg = self.INCONCLUSIVE_MESSAGES.get(self._stage_idx, "Results inconclusive. Retry needed.")
                return StepResult(state=msg, done=False)

            self._stage_idx += 1
            stage = self.STAGES[self._stage_idx]
            done = stage == "treatment_prescribed"
            return StepResult(state=self.STATE_DESCRIPTIONS[stage], done=done)
        return StepResult(
            state=self._state + f" (You tried '{action}' but it was not the appropriate next step.)",
            done=False,
        )


# ======================================================================
# Registry
# ======================================================================

# Environments used for training trajectory generation
# v2: expanded from 3 to 10 for better generalization
# Includes linear, branching, and stochastic environments
TRAIN_ENVIRONMENTS: list[type[TextEnvironment]] = [
    DocumentWorkflow,
    CodeReview,
    EmailTriage,
    DataPipeline,
    ResearchTask,
    BugTriage,
    OnboardingProcess,
    SecurityAudit,
    ProjectPlanning,      # branching
    NoisyDeployment,      # stochastic
]

# Environments used ONLY for evaluation (never seen during training)
TEST_ENVIRONMENTS: list[type[TextEnvironment]] = [
    CustomerSupport,
    IncidentResponse,
    MeetingPreparation,
    ContentPublishing,
    ExperimentPipeline,
    TroubleshootingGuide,  # branching (OOD)
    UncertainDiagnosis,    # stochastic (OOD)
]

# All environments
ALL_ENVIRONMENTS: list[type[TextEnvironment]] = TRAIN_ENVIRONMENTS + TEST_ENVIRONMENTS


def make_train(seed: int = 0) -> list[TextEnvironment]:
    """Instantiate training environments only."""
    return [cls(seed=seed) for cls in TRAIN_ENVIRONMENTS]


def make_test(seed: int = 0) -> list[TextEnvironment]:
    """Instantiate test-only environments (unseen during training)."""
    return [cls(seed=seed) for cls in TEST_ENVIRONMENTS]


def make_all(seed: int = 0) -> list[TextEnvironment]:
    """Instantiate one of each environment with the given seed."""
    return [cls(seed=seed) for cls in ALL_ENVIRONMENTS]
