"""
Deterministic Pushdown Automaton (DPDA) for grammar-constrained generation.

Implements a predictive LL(1) parser that drives generation token-by-token.
At each step, the PDA exposes the set of valid token IDs (via ``get_tokens``)
and updates its state after a token is generated (via ``next_state``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PDAError(Exception):
    """Base exception for PDA-related errors."""

    pass


class AmbiguousTokenError(PDAError):
    """Raised when a generated token maps to multiple possible terminals."""

    pass


class UnexpectedTokenError(PDAError):
    """Raised when a generated token does not match the expected terminal."""

    pass


class PushdownAutomaton:
    """Deterministic Pushdown Automaton for LL(1) grammar-constrained decoding.

    Attributes:
        stack: The parsing stack (top = last element).
        start_symbol: The initial grammar symbol (used by ``reset()``).
        grammar: The LL(1) parsing table.
        map_terminals_tokens: Mapping from terminal names to lists of token IDs.
        map_tokens_terminals: Inverse mapping from token IDs to terminal names.
        current_terminals: Terminals reachable from the current stack state.
    """

    def __init__(
        self,
        grammar: dict[str, Any],
        startSymbol: str,
        map: dict[str, Any],
    ) -> None:
        """Initialize the PDA.

        Args:
            grammar: The LL(1) parsing table. Keys are non-terminal names;
                values are dicts mapping terminal names to production rules.
            startSymbol: The start symbol of the grammar (e.g. ``"S*"``).
            map: Mapping from non-terminal / terminal names to lists of
                token IDs.  Can be a flat dict ``{terminal: [ids]}`` or a
                nested dict ``{non_terminal: {terminal: [ids]}}``.
        """
        self.stack: list[str] = [startSymbol]
        self.start_symbol: str = startSymbol
        self.grammar: dict[str, Any] = grammar
        self.map_terminals_tokens: dict[str, Any] = map
        self.map_tokens_terminals: dict[int, list[str]] = {}
        self.current_terminals: list[str] = []

        # Build inverse mapping: token_id → [terminal, ...]
        for non_terminal, value in map.items():
            if isinstance(value, dict):
                for terminal, tokens in value.items():
                    if isinstance(tokens, list):
                        for token in tokens:
                            self.map_tokens_terminals.setdefault(
                                token, []
                            ).append(terminal)
            elif isinstance(value, list):
                for token in value:
                    self.map_tokens_terminals.setdefault(
                        token, []
                    ).append(non_terminal)

    def reset(self) -> None:
        """Reset the automaton to its initial state."""
        self.stack = [self.start_symbol]
        self.current_terminals = []
        logger.info("PDA reset: stack = %s", self.stack)

    # ------------------------------------------------------------------
    # Token discovery
    # ------------------------------------------------------------------

    def recursive_get_tokens(
        self,
        stack: list[str],
        visited: set[str] | None = None,
    ) -> list[str]:
        """Recursively explore the stack to find all reachable terminals.

        Uses a ``visited`` set to prevent infinite recursion on cyclic
        grammars.  When a previously-visited non-terminal is encountered,
        it is skipped (no further expansion), which is safe for LL(1)
        grammars without left-recursion.

        Args:
            stack: The current parser stack (mutated during recursion).
            visited: Set of already-visited non-terminals.

        Returns:
            List of terminal names reachable from the current stack state.
        """
        if visited is None:
            visited = set()

        if not stack:
            return []

        top = stack.pop()

        if top in visited:
            return []

        visited.add(top)

        if top not in self.grammar:
            # Terminal — return it directly
            return [top]

        tokens: list[str] = []
        for symbol in self.grammar[top]:
            if symbol not in visited:
                stack.extend(reversed([symbol]))
                tokens += self.recursive_get_tokens(stack, visited)

        return tokens

    def get_tokens(self) -> list[int]:
        """Return the list of token IDs valid for the next generation step.

        Explores the stack recursively to find all reachable terminals,
        then maps them to token IDs.  Verifies that token sets for
        different terminals are disjoint.

        Returns:
            List of valid token IDs for the next step.

        Raises:
            ValueError: If token sets for different terminals overlap.
        """
        terminals = self.recursive_get_tokens(self.stack.copy())
        tokens: set[int] = set()

        for terminal in terminals:
            terminal_tokens = set(self.map_terminals_tokens[terminal])
            if tokens and not terminal_tokens.isdisjoint(tokens):
                raise ValueError(
                    f"Token sets associated with terminals are not disjoint. "
                    f"Terminal '{terminal}' shares tokens with another terminal. "
                    f"Intersection: {terminal_tokens & tokens}"
                )
            tokens.update(terminal_tokens)

        self.current_terminals = terminals
        return list(tokens)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def next_state(self, token_gen: int) -> None:
        """Transition the PDA to the next state given a generated token.

        Args:
            token_gen: The token ID that was generated.

        Raises:
            AttributeError: If ``current_terminals`` has not been populated
                (``get_tokens()`` was not called before this method).
            AmbiguousTokenError: If the token maps to more than one
                possible terminal in the current state.
            IndexError: If the token does not map to any terminal.
        """
        logger.info("Current terminals: %s", self.current_terminals)

        if token_gen not in self.map_tokens_terminals:
            raise PDAError(
                f"Token {token_gen} not found in token-to-terminal map."
            )

        check_terminals = set(self.map_tokens_terminals[token_gen]).intersection(
            set(self.current_terminals)
        )
        logger.info("Matching terminals: %s", check_terminals)

        if len(check_terminals) == 0:
            raise PDAError(
                f"Token {token_gen} does not match any current terminal. "
                f"Current terminals: {self.current_terminals}"
            )
        if len(check_terminals) > 1:
            raise AmbiguousTokenError(
                f"Token {token_gen} is ambiguous: it corresponds to "
                f"multiple possible terminals {check_terminals} "
                f"for the current state."
            )

        terminal = list(check_terminals)[0]
        self.next_state_terminal(terminal)

    def next_state_terminal(self, terminal: str) -> None:
        """Update the stack based on a matched terminal.

        Pops the top of the stack.  If it is a non-terminal, expands it
        using the grammar table.  If it is a terminal, verifies it matches.

        Args:
            terminal: The terminal name that was matched.

        Raises:
            UnexpectedTokenError: If the top-of-stack terminal does not
                match the expected terminal.
        """
        token = terminal
        stack = self.stack
        top = stack.pop()

        if top in self.grammar:
            # Non-terminal at top — expand it
            for symbol in reversed(self.grammar[top][token]):
                stack.append(symbol)

            # Recurse to handle the terminal
            self.next_state_terminal(token)
            return

        # Terminal at top — verify match
        if top != token:
            logger.error(
                "Parser stack: %s | Comparing: %r vs %r",
                stack,
                top,
                token,
            )
            raise UnexpectedTokenError(
                f"Unexpected token: found {top!r}, expected {token!r}."
            )

    def eos(self) -> bool:
        """Return ``True`` if the stack is empty (end-of-string)."""
        return not bool(self.stack)
