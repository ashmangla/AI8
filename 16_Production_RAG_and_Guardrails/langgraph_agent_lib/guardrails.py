"""Guardrails integration for production-safe LangGraph agents.

This module provides utilities for integrating Guardrails AI validation
into LangGraph agent workflows, including input and output validation.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable
from typing_extensions import TypedDict, Annotated

from guardrails.hub import (
    RestrictToTopic,
    DetectJailbreak,
    CompetitorCheck,
    LlmRagEvaluator,
    HallucinationPrompt,
    ProfanityFree,
    GuardrailsPII
)
from guardrails import Guard
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# Set up logging
logger = logging.getLogger(__name__)


def _normalize_validation_results(
    existing_results: Optional[Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return a dict-based structure and mutable guardrails list."""

    if isinstance(existing_results, dict):
        base_results = dict(existing_results)
        guardrails_results = list(existing_results.get("guardrails", []))
    elif isinstance(existing_results, list):
        guardrails_results = list(existing_results)
        base_results = {"guardrails": guardrails_results.copy()}
    else:
        base_results = {}
        guardrails_results = []

    return base_results, guardrails_results


async def _run_guard_validation_async(func: Callable[..., Dict[str, Any]], *args) -> Dict[str, Any]:
    """Execute blocking guard validation without blocking the event loop.

    Falls back to synchronous execution if the executor task is cancelled.
    """

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.shield(loop.run_in_executor(None, lambda: func(*args)))
    except asyncio.CancelledError as exc:
        logger.warning(
            "Guard validation executor cancelled; retrying synchronously",
            exc_info=True,
        )
        return func(*args)


class GuardrailsState(TypedDict):
    """State schema for guardrails-enabled agent graphs.
    
    Attributes:
        messages: List of messages in the conversation history.
        validation_results: Optional validation results from guardrails.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Optional[Dict[str, Any]]


def create_guardrails_guard(
    valid_topics: Optional[List[str]] = None,
    invalid_topics: Optional[List[str]] = None,
    enable_jailbreak_detection: bool = True,
    enable_pii_protection: bool = True,
    enable_profanity_check: bool = True,
    enable_competitor_check: bool = False,
    pii_entities: Optional[List[str]] = None
) -> Guard:
    """Create a Guardrails guard with common production safety checks.
    
    Args:
        valid_topics: List of valid topics to allow. None disables topic restriction.
        invalid_topics: List of invalid topics to block. None disables topic restriction.
        enable_jailbreak_detection: Whether to enable jailbreak detection. Default: True.
        enable_pii_protection: Whether to enable PII detection and redaction. Default: True.
        enable_profanity_check: Whether to enable profanity filtering. Default: True.
        enable_competitor_check: Whether to enable competitor mention detection. Default: False.
        pii_entities: List of PII entity types to detect. Default: Common PII types.
        
    Returns:
        Configured Guard instance.
        
    Raises:
        RuntimeError: If guard configuration fails.
    """
    guard = Guard()
    
    try:
        # Topic restriction
        if valid_topics or invalid_topics:
            guard = guard.use(
                RestrictToTopic(
                    valid_topics=valid_topics or [],
                    invalid_topics=invalid_topics or [],
                    disable_classifier=True,
                    disable_llm=False,
                    on_fail="exception"
                )
            )
            logger.debug("Topic restriction guard configured")
        
        # Jailbreak detection
        if enable_jailbreak_detection:
            guard = guard.use(DetectJailbreak())
            logger.debug("Jailbreak detection guard configured")
        
        # PII protection
        if enable_pii_protection:
            default_entities = ["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]
            entities = pii_entities or default_entities
            guard = guard.use(
                GuardrailsPII(
                    entities=entities,
                    on_fail="fix"
                )
            )
            logger.debug(f"PII protection guard configured for entities: {entities}")
        
        # Profanity check
        if enable_profanity_check:
            guard = guard.use(
                ProfanityFree(
                    threshold=0.8,
                    validation_method="sentence",
                    on_fail="exception"
                )
            )
            logger.debug("Profanity check guard configured")
        
        # Competitor check (optional)
        if enable_competitor_check:
            guard = guard.use(CompetitorCheck())
            logger.debug("Competitor check guard configured")
        
        logger.info("Guardrails guard configured successfully")
        return guard
        
    except Exception as e:
        logger.error(f"Failed to configure guardrails: {e}", exc_info=True)
        raise RuntimeError(f"Failed to configure guardrails: {e}") from e


def create_factuality_guard(
    eval_model: str = "gpt-4.1-mini",
    on_prompt: bool = True
) -> Guard:
    """Create a factuality guard for RAG responses.
    
    Args:
        eval_model: Model to use for factuality evaluation. Default: "gpt-4.1-mini".
        on_prompt: Whether to validate at prompt stage or response stage. Default: True.
        
    Returns:
        Configured Guard instance for factuality checking.
        
    Raises:
        RuntimeError: If guard configuration fails.
    """
    try:
        guard = Guard().use(
            LlmRagEvaluator(
                eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                llm_evaluator_fail_response="hallucinated",
                llm_evaluator_pass_response="factual",
                llm_callable=eval_model,
                on_fail="exception",
                on="prompt" if on_prompt else "response"
            )
        )
        logger.info(f"Factuality guard configured with model: {eval_model}")
        return guard
    except Exception as e:
        logger.error(f"Failed to configure factuality guard: {e}", exc_info=True)
        raise RuntimeError(f"Failed to configure factuality guard: {e}") from e


def validate_input(
    guard: Guard,
    user_input: str,
    raise_on_failure: bool = True
) -> Dict[str, Any]:
    """Validate user input using a Guardrails guard.
    
    Args:
        guard: The Guard instance to use for validation.
        user_input: The user input to validate.
        raise_on_failure: Whether to raise an exception on validation failure.
            If False, returns validation result. Default: True.
        
    Returns:
        Dictionary with validation results including:
        - validation_passed: Boolean indicating if validation passed
        - validated_output: The validated (and potentially modified) output
        - error: Error message if validation failed
        
    Raises:
        RuntimeError: If validation fails and raise_on_failure is True.
    """
    try:
        result = guard.validate(user_input)
        
        validation_result = {
            "validation_passed": result.validation_passed,
            "validated_output": result.validated_output if hasattr(result, 'validated_output') else user_input,
            "error": None
        }
        
        if not result.validation_passed and raise_on_failure:
            error_msg = f"Input validation failed: {getattr(result, 'error', 'Unknown error')}"
            logger.warning(f"Input validation failed: {user_input[:100]}...")
            raise RuntimeError(error_msg)
        
        return validation_result
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Input validation error: {e}", exc_info=True)
        if raise_on_failure:
            raise RuntimeError(f"Input validation failed: {e}") from e
        return {
            "validation_passed": False,
            "validated_output": user_input,
            "error": str(e)
        }


def validate_output(
    guard: Guard,
    agent_response: str,
    context: Optional[str] = None,
    raise_on_failure: bool = True
) -> Dict[str, Any]:
    """Validate agent output using a Guardrails guard.
    
    Args:
        guard: The Guard instance to use for validation.
        agent_response: The agent's response to validate.
        context: Optional context for factuality checking.
        raise_on_failure: Whether to raise an exception on validation failure.
            If False, returns validation result. Default: True.
        
    Returns:
        Dictionary with validation results.
        
    Raises:
        RuntimeError: If validation fails and raise_on_failure is True.
    """
    try:
        # For factuality guards, include context if provided
        if context:
            result = guard.validate(agent_response, metadata={"context": context})
        else:
            result = guard.validate(agent_response)
        
        validation_result = {
            "validation_passed": result.validation_passed,
            "validated_output": result.validated_output if hasattr(result, 'validated_output') else agent_response,
            "error": None
        }
        
        if not result.validation_passed and raise_on_failure:
            error_msg = f"Output validation failed: {getattr(result, 'error', 'Unknown error')}"
            logger.warning(f"Output validation failed: {agent_response[:100]}...")
            raise RuntimeError(error_msg)
        
        return validation_result
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Output validation error: {e}", exc_info=True)
        if raise_on_failure:
            raise RuntimeError(f"Output validation failed: {e}") from e
        return {
            "validation_passed": False,
            "validated_output": agent_response,
            "error": str(e)
        }


def create_guardrails_node(
    input_guard: Optional[Guard] = None,
    output_guard: Optional[Guard] = None,
    strict_mode: bool = True
):
    """Create a LangGraph node that validates inputs and outputs with Guardrails.
    
    Args:
        input_guard: Guard for validating user inputs. If None, input validation is skipped.
        output_guard: Guard for validating agent outputs. If None, output validation is skipped.
        strict_mode: If True, raises exceptions on validation failure.
            If False, logs warnings but continues. Default: True.
        
    Returns:
        A function that can be used as a LangGraph node.
    """
    def guardrails_node(state: GuardrailsState) -> Dict[str, Any]:
        """Validate messages in the agent state.
        
        Args:
            state: Current agent state with messages.
            
        Returns:
            Updated state with validation results.
        """
        messages = state.get("messages", [])
        base_results, guardrails_results = _normalize_validation_results(
            state.get("validation_results")
        )
        
        if not messages:
            base_results.setdefault("guardrails", guardrails_results)
            return {"validation_results": base_results}
        
        # Validate the last message
        last_message = messages[-1]
        
        try:
            if isinstance(last_message, HumanMessage) and input_guard:
                logger.info(
                    "Guardrails (input) validation started | snippet='%s'",
                    last_message.content[:100].replace("\n", " ") if last_message.content else "",
                )
                # Validate user input
                logger.debug("Validating user input with guardrails")
                result = validate_input(
                    input_guard,
                    last_message.content,
                    raise_on_failure=strict_mode
                )
                guardrails_results.append({
                    "type": "input",
                    "passed": result["validation_passed"],
                    "message": last_message.content[:100]
                })

                logger.info(
                    "Guardrails (input) validation finished | passed=%s",
                    result["validation_passed"],
                )
                
                # If validation modified the input, we could update the message here
                if not result["validation_passed"] and strict_mode:
                    logger.error(f"Input validation failed: {result.get('error')}")
            
            elif isinstance(last_message, AIMessage) and output_guard:
                logger.info(
                    "Guardrails (output) validation started | snippet='%s'",
                    last_message.content[:100].replace("\n", " ") if last_message.content else "",
                )
                # Validate agent output
                logger.debug("Validating agent output with guardrails")
                result = validate_output(
                    output_guard,
                    last_message.content,
                    raise_on_failure=strict_mode
                )
                guardrails_results.append({
                    "type": "output",
                    "passed": result["validation_passed"],
                    "message": last_message.content[:100]
                })

                logger.info(
                    "Guardrails (output) validation finished | passed=%s",
                    result["validation_passed"],
                )
                
                if not result["validation_passed"] and strict_mode:
                    logger.error(f"Output validation failed: {result.get('error')}")
                    
        except Exception as e:
            logger.error(f"Guardrails validation error: {e}", exc_info=True)
            if strict_mode:
                raise
            guardrails_results.append({
                "type": "error",
                "passed": False,
                "error": str(e)
            })
        else:
            if guardrails_results:
                logger.info(
                    "Guardrails validation summary | results=%s",
                    [
                        {
                            "type": item.get("type"),
                            "passed": item.get("passed"),
                        }
                        for item in guardrails_results
                    ],
                )

        base_results["guardrails"] = guardrails_results
        return {"validation_results": base_results}
    
    return guardrails_node


async def validate_input_async(
    guard: Guard,
    user_input: str,
    raise_on_failure: bool = True,
) -> Dict[str, Any]:
    """Async wrapper around validate_input using a thread executor."""
    return await _run_guard_validation_async(
        validate_input,
        guard,
        user_input,
        raise_on_failure,
    )


async def validate_output_async(
    guard: Guard,
    agent_response: str,
    context: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Dict[str, Any]:
    """Async wrapper around validate_output using a thread executor."""
    return await _run_guard_validation_async(
        validate_output,
        guard,
        agent_response,
        context,
        raise_on_failure,
    )


def create_guardrails_node_async(
    input_guard: Optional[Guard] = None,
    output_guard: Optional[Guard] = None,
    strict_mode: bool = True,
):
    """Async variant of create_guardrails_node for non-blocking guard validation."""

    async def guardrails_node(state: GuardrailsState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        base_results, guardrails_results = _normalize_validation_results(
            state.get("validation_results")
        )

        if not messages:
            base_results.setdefault("guardrails", guardrails_results)
            return {"validation_results": base_results}

        last_message = messages[-1]

        try:
            if isinstance(last_message, HumanMessage) and input_guard:
                logger.info(
                    "Guardrails (async input) validation started | snippet='%s'",
                    last_message.content[:100].replace("\n", " ") if last_message.content else "",
                )
                result = await validate_input_async(
                    input_guard,
                    last_message.content,
                    raise_on_failure=strict_mode,
                )
                guardrails_results.append(
                    {
                        "type": "input",
                        "passed": result["validation_passed"],
                        "message": last_message.content[:100],
                    }
                )
                logger.info(
                    "Guardrails (async input) validation finished | passed=%s",
                    result["validation_passed"],
                )
                if not result["validation_passed"] and strict_mode:
                    logger.error(f"Async input validation failed: {result.get('error')}")

            elif isinstance(last_message, AIMessage) and output_guard:
                logger.info(
                    "Guardrails (async output) validation started | snippet='%s'",
                    last_message.content[:100].replace("\n", " ") if last_message.content else "",
                )
                result = await validate_output_async(
                    output_guard,
                    last_message.content,
                    raise_on_failure=strict_mode,
                )
                guardrails_results.append(
                    {
                        "type": "output",
                        "passed": result["validation_passed"],
                        "message": last_message.content[:100],
                    }
                )
                logger.info(
                    "Guardrails (async output) validation finished | passed=%s",
                    result["validation_passed"],
                )
                if not result["validation_passed"] and strict_mode:
                    logger.error(f"Async output validation failed: {result.get('error')}")

        except Exception as exc:
            logger.error("Guardrails async validation error: %s", exc, exc_info=True)
            if strict_mode:
                raise
            guardrails_results.append(
                {
                    "type": "error",
                    "passed": False,
                    "error": str(exc),
                }
            )

        if guardrails_results:
            logger.info(
                "Guardrails async validation summary | results=%s",
                [
                    {
                        "type": item.get("type"),
                        "passed": item.get("passed"),
                    }
                    for item in guardrails_results
                ],
            )

        base_results["guardrails"] = guardrails_results
        return {"validation_results": base_results}

    return guardrails_node

