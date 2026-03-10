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
# Registry
# ======================================================================

# Environments used for training trajectory generation
TRAIN_ENVIRONMENTS: list[type[TextEnvironment]] = [
    DocumentWorkflow,
    CodeReview,
    EmailTriage,
]

# Environments used ONLY for evaluation (never seen during training)
TEST_ENVIRONMENTS: list[type[TextEnvironment]] = [
    CustomerSupport,
    IncidentResponse,
    MeetingPreparation,
]

# All environments (for backward compat)
ALL_ENVIRONMENTS: list[type[TextEnvironment]] = TRAIN_ENVIRONMENTS + [
    DataPipeline,
    ResearchTask,
] + TEST_ENVIRONMENTS


def make_train(seed: int = 0) -> list[TextEnvironment]:
    """Instantiate training environments only."""
    return [cls(seed=seed) for cls in TRAIN_ENVIRONMENTS]


def make_test(seed: int = 0) -> list[TextEnvironment]:
    """Instantiate test-only environments (unseen during training)."""
    return [cls(seed=seed) for cls in TEST_ENVIRONMENTS]


def make_all(seed: int = 0) -> list[TextEnvironment]:
    """Instantiate one of each environment with the given seed."""
    return [cls(seed=seed) for cls in ALL_ENVIRONMENTS]
