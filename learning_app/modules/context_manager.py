"""
Context Manager Module for RAG Chat.

Manages token budgets and context window constraints:
- Token counting and budget allocation
- Context truncation strategies
- History pruning for multi-turn conversations
- Educational explanations of context management

Core concept: LLMs have limited context windows. We must fit:
- System prompt (instructions)
- Retrieved documents (RAG context)
- Conversation history
- Room for generation
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TokenBudget:
    """Token budget allocation for RAG pipeline."""

    total_tokens: int          # Total available context window
    system_tokens: int         # Tokens for system prompt
    context_tokens: int        # Tokens for retrieved documents
    history_tokens: int        # Tokens for conversation history
    generation_tokens: int     # Tokens reserved for response
    used_tokens: int = 0       # Tokens actually used

    def get_allocation_report(self) -> str:
        """
        Get human-readable budget allocation report.

        Returns:
            Formatted string showing token distribution
        """
        return f"""
Token Budget Allocation:
  Total Context Window: {self.total_tokens:,} tokens
  ├─ System Prompt:      {self.system_tokens:,} tokens ({self._percent(self.system_tokens)})
  ├─ RAG Context:        {self.context_tokens:,} tokens ({self._percent(self.context_tokens)})
  ├─ Conv. History:      {self.history_tokens:,} tokens ({self._percent(self.history_tokens)})
  └─ Generation Buffer:  {self.generation_tokens:,} tokens ({self._percent(self.generation_tokens)})

  Available for input:   {self.system_tokens + self.context_tokens + self.history_tokens:,} tokens
  Used so far:          {self.used_tokens:,} tokens
        """.strip()

    def _percent(self, tokens: int) -> str:
        """Calculate percentage of total."""
        return f"{(tokens / self.total_tokens * 100):.1f}%"


class ContextManager:
    """
    Manages context window and token budgets for RAG chat.

    Key responsibilities:
    1. Calculate token budgets based on model context window
    2. Truncate contexts to fit within budgets
    3. Prune conversation history intelligently
    4. Provide educational explanations of constraints

    Educational focus: Shows how production RAG systems handle
    limited context windows and make trade-offs.
    """

    def __init__(
        self,
        context_window: int = 1024,
        system_prompt_ratio: float = 0.1,
        context_ratio: float = 0.5,
        history_ratio: float = 0.2,
        generation_ratio: float = 0.2
    ):
        """
        Initialize context manager.

        Args:
            context_window: Total tokens available
            system_prompt_ratio: Fraction for system prompt
            context_ratio: Fraction for RAG context
            history_ratio: Fraction for conversation history
            generation_ratio: Fraction reserved for generation

        Note: Ratios should sum to 1.0
        """
        self.context_window = context_window

        # Validate ratios
        total_ratio = (
            system_prompt_ratio +
            context_ratio +
            history_ratio +
            generation_ratio
        )
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(
                f"Budget ratios must sum to 1.0, got {total_ratio}"
            )

        # Calculate token budgets
        self.budget = TokenBudget(
            total_tokens=context_window,
            system_tokens=int(context_window * system_prompt_ratio),
            context_tokens=int(context_window * context_ratio),
            history_tokens=int(context_window * history_ratio),
            generation_tokens=int(context_window * generation_ratio)
        )

    def allocate_budget(
        self,
        system_prompt: str,
        rag_context: str,
        conversation_history: List[Dict[str, str]],
        token_counter
    ) -> Tuple[str, str, List[Dict[str, str]], Dict]:
        """
        Allocate token budget and truncate if needed.

        Args:
            system_prompt: System instructions
            rag_context: Retrieved document context
            conversation_history: List of {role, content} messages
            token_counter: Function to count tokens

        Returns:
            Tuple of (truncated_system, truncated_context,
                     truncated_history, allocation_stats)
        """
        # Count tokens
        system_tokens = token_counter(system_prompt)
        context_tokens = token_counter(rag_context)
        history_tokens = self._count_history_tokens(
            conversation_history,
            token_counter
        )

        # Track stats
        stats = {
            "original": {
                "system": system_tokens,
                "context": context_tokens,
                "history": history_tokens,
                "total": system_tokens + context_tokens + history_tokens
            },
            "budget": {
                "system": self.budget.system_tokens,
                "context": self.budget.context_tokens,
                "history": self.budget.history_tokens
            },
            "truncated": False,
            "truncation_details": []
        }

        # Truncate if needed
        truncated_system = self._truncate_text(
            system_prompt,
            self.budget.system_tokens,
            token_counter,
            "System prompt"
        )
        if truncated_system != system_prompt:
            stats["truncated"] = True
            stats["truncation_details"].append("system_prompt")

        truncated_context = self._truncate_text(
            rag_context,
            self.budget.context_tokens,
            token_counter,
            "RAG context"
        )
        if truncated_context != rag_context:
            stats["truncated"] = True
            stats["truncation_details"].append("rag_context")

        truncated_history = self._truncate_history(
            conversation_history,
            self.budget.history_tokens,
            token_counter
        )
        if len(truncated_history) < len(conversation_history):
            stats["truncated"] = True
            stats["truncation_details"].append("conversation_history")

        # Update final token counts
        stats["final"] = {
            "system": token_counter(truncated_system),
            "context": token_counter(truncated_context),
            "history": self._count_history_tokens(
                truncated_history,
                token_counter
            ),
            "total": (
                token_counter(truncated_system) +
                token_counter(truncated_context) +
                self._count_history_tokens(truncated_history, token_counter)
            )
        }

        # Update used tokens in budget
        self.budget.used_tokens = stats["final"]["total"]

        return (
            truncated_system,
            truncated_context,
            truncated_history,
            stats
        )

    def _truncate_text(
        self,
        text: str,
        max_tokens: int,
        token_counter,
        label: str
    ) -> str:
        """
        Truncate text to fit token budget.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            token_counter: Function to count tokens
            label: Description for logging

        Returns:
            Truncated text
        """
        tokens = token_counter(text)

        if tokens <= max_tokens:
            return text

        # Truncate by characters (rough estimate: 1 token ≈ 4 chars)
        target_chars = int(max_tokens * 4)

        # Keep beginning and add truncation notice
        truncated = text[:target_chars]
        truncated += f"\n\n[... {label} truncated to fit budget ...]"

        return truncated

    def _truncate_history(
        self,
        history: List[Dict[str, str]],
        max_tokens: int,
        token_counter
    ) -> List[Dict[str, str]]:
        """
        Truncate conversation history to fit budget.

        Strategy: Keep most recent messages first.

        Args:
            history: List of messages
            max_tokens: Maximum tokens allowed
            token_counter: Function to count tokens

        Returns:
            Truncated history (most recent messages)
        """
        if not history:
            return []

        truncated = []
        tokens_used = 0

        # Iterate from most recent to oldest
        for message in reversed(history):
            message_tokens = token_counter(message["content"])

            if tokens_used + message_tokens <= max_tokens:
                truncated.insert(0, message)  # Add at beginning
                tokens_used += message_tokens
            else:
                break

        return truncated

    def _count_history_tokens(
        self,
        history: List[Dict[str, str]],
        token_counter
    ) -> int:
        """Count total tokens in conversation history."""
        return sum(token_counter(msg["content"]) for msg in history)

    def format_allocation_report(
        self,
        allocation_stats: Dict
    ) -> str:
        """
        Format allocation statistics as human-readable report.

        Args:
            allocation_stats: Stats from allocate_budget()

        Returns:
            Formatted HTML report
        """
        orig = allocation_stats["original"]
        final = allocation_stats["final"]
        budget = allocation_stats["budget"]
        truncated = allocation_stats["truncated"]

        html = f"""
        <div style="font-family: sans-serif; padding: 15px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 8px;">
            <h4 style="margin-top: 0;">📊 Token Budget Allocation</h4>

            <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                <tr style="background: var(--background-fill-secondary); font-weight: bold;">
                    <th style="padding: 8px; text-align: left;">Component</th>
                    <th style="padding: 8px; text-align: right;">Original</th>
                    <th style="padding: 8px; text-align: right;">Budget</th>
                    <th style="padding: 8px; text-align: right;">Final</th>
                    <th style="padding: 8px; text-align: left;">Status</th>
                </tr>
                <tr>
                    <td style="padding: 8px;">System Prompt</td>
                    <td style="padding: 8px; text-align: right;">{orig['system']:,}</td>
                    <td style="padding: 8px; text-align: right;">{budget['system']:,}</td>
                    <td style="padding: 8px; text-align: right;">{final['system']:,}</td>
                    <td style="padding: 8px;">{self._status_badge(orig['system'], budget['system'])}</td>
                </tr>
                <tr style="background: var(--background-fill-secondary);">
                    <td style="padding: 8px;">RAG Context</td>
                    <td style="padding: 8px; text-align: right;">{orig['context']:,}</td>
                    <td style="padding: 8px; text-align: right;">{budget['context']:,}</td>
                    <td style="padding: 8px; text-align: right;">{final['context']:,}</td>
                    <td style="padding: 8px;">{self._status_badge(orig['context'], budget['context'])}</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Conv. History</td>
                    <td style="padding: 8px; text-align: right;">{orig['history']:,}</td>
                    <td style="padding: 8px; text-align: right;">{budget['history']:,}</td>
                    <td style="padding: 8px; text-align: right;">{final['history']:,}</td>
                    <td style="padding: 8px;">{self._status_badge(orig['history'], budget['history'])}</td>
                </tr>
                <tr style="background: var(--background-fill-secondary); font-weight: bold;">
                    <td style="padding: 8px;">Total Input</td>
                    <td style="padding: 8px; text-align: right;">{orig['total']:,}</td>
                    <td style="padding: 8px; text-align: right;">{budget['system'] + budget['context'] + budget['history']:,}</td>
                    <td style="padding: 8px; text-align: right;">{final['total']:,}</td>
                    <td style="padding: 8px;">{self._status_badge(orig['total'], budget['system'] + budget['context'] + budget['history'])}</td>
                </tr>
            </table>

            <div style="margin-top: 10px; padding: 10px; background: {'rgba(251, 191, 36, 0.2)' if truncated else 'rgba(34, 197, 94, 0.2)'}; border-radius: 5px;">
                <strong>{'⚠️ Truncation Applied' if truncated else '✓ Fits within budget'}</strong>
                {f"<br><small>Truncated: {', '.join(allocation_stats['truncation_details'])}</small>" if truncated else ""}
            </div>

            <div style="margin-top: 10px; font-size: 0.9em; opacity: 0.8;">
                <strong>Context Window:</strong> {self.context_window:,} tokens<br>
                <strong>Available for generation:</strong> {self.budget.generation_tokens:,} tokens<br>
                <strong>Remaining budget:</strong> {self.context_window - final['total'] - self.budget.generation_tokens:,} tokens
            </div>
        </div>
        """

        return html

    def _status_badge(self, original: int, budget: int) -> str:
        """Generate status badge HTML."""
        if original <= budget:
            return '<span style="color: rgb(34, 197, 94);">✓ OK</span>'
        else:
            return '<span style="color: rgb(251, 191, 36);">⚠ Truncated</span>'

    def get_budget_info(self) -> Dict:
        """
        Get current budget information.

        Returns:
            Dictionary with budget details
        """
        return {
            "context_window": self.context_window,
            "budgets": {
                "system": self.budget.system_tokens,
                "context": self.budget.context_tokens,
                "history": self.budget.history_tokens,
                "generation": self.budget.generation_tokens
            },
            "used_tokens": self.budget.used_tokens,
            "available_tokens": self.context_window - self.budget.used_tokens
        }
