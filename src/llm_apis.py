#@title `LanguageModelAPI`
import asyncio
import os
import time
import logging
from tqdm.auto import tqdm
from abc import ABC
from abc import abstractmethod
from tqdm.asyncio import tqdm as async_tqdm
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict, Counter

from google import genai
from google.genai.types import GenerateContentConfig
from openai import AsyncOpenAI

from utils import validate_genai_response_constraint

async def _run_and_return_index(index, coro):
    try:
        result = await coro
        return index, result
    except Exception as e:
        return index, e

class BatchMetrics:
    """Class to track batch processing metrics and errors."""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.retry_counts = Counter()
        self.completed_requests = 0
        self.total_errors = 0
        self.active_requests = 0
        self.start_time = time.time()
        self.error_details = []
        self.success_details = []
        
    def add_error(self, error_type: str, error_msg: str, prompt_index: int, attempt: int):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        self.total_errors += 1
        self.error_details.append({
            'type': error_type,
            'message': error_msg,
            'prompt_index': prompt_index,
            'attempt': attempt,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_success(self, prompt_index: int, attempts_taken: int):
        """Record a successful completion."""
        self.completed_requests += 1
        self.retry_counts[attempts_taken] += 1
        self.success_details.append({
            'prompt_index': prompt_index,
            'attempts': attempts_taken,
            'timestamp': datetime.now().isoformat()
        })
    
    def increment_active(self):
        """Increment active request counter."""
        self.active_requests += 1
    
    def decrement_active(self):
        """Decrement active request counter."""
        self.active_requests = max(0, self.active_requests - 1)
    
    def get_tqdm_postfix(self) -> Dict[str, Any]:
        """Get postfix dictionary for tqdm progress bar."""
        return {
            'errors': self.total_errors,
            'active': self.active_requests,
            'completed': self.completed_requests,
            '429s': self.error_counts.get('rate_limit', 0),
            '503s': self.error_counts.get('service_unavailable', 0)
        }
    
    def print_summary_report(self, total_prompts: int, logger: logging.Logger):
        """Print comprehensive batch processing report."""
        duration = time.time() - self.start_time
        success_rate = (self.completed_requests / total_prompts) * 100 if total_prompts > 0 else 0
        
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING SUMMARY REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration:.2f} seconds")
        logger.info(f"Total Prompts: {total_prompts}")
        logger.info(f"Successful: {self.completed_requests} ({success_rate:.1f}%)")
        logger.info(f"Failed: {total_prompts - self.completed_requests}")
        logger.info(f"Total Error Occurrences: {self.total_errors}")
        
        if self.error_counts:
            logger.info("\nERROR BREAKDOWN:")
            for error_type, count in sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.total_errors) * 100
                logger.info(f"  {error_type}: {count} ({percentage:.1f}%)")
        
        if self.retry_counts:
            logger.info("\nRETRY STATISTICS:")
            for attempts, count in sorted(self.retry_counts.items()):
                percentage = (count / self.completed_requests) * 100 if self.completed_requests > 0 else 0
                logger.info(f"  {attempts} attempt(s): {count} requests ({percentage:.1f}%)")
            
            avg_attempts = sum(attempts * count for attempts, count in self.retry_counts.items()) / self.completed_requests if self.completed_requests > 0 else 0
            logger.info(f"  Average attempts per successful request: {avg_attempts:.2f}")
        
        if duration > 0:
            logger.info(f"\nTHROUGHPUT:")
            logger.info(f"  Requests per second: {self.completed_requests / duration:.2f}")
            logger.info(f"  Average time per request: {duration / total_prompts:.2f} seconds")
        
        logger.info("=" * 60)

class LanguageModelAPI(ABC):
    """
    An abstract base class for interacting with various Language Model APIs.

    This class provides a standardized interface for making single and batch API
    calls, with built-in support for asynchronous operations, parallelization,
    logging, and history tracking. Subclasses must implement the backend-specific
    logic for making API calls and handling their responses.

    Attributes:
        model_name (str): The specific model to be used for the API calls.
        api_key (Optional[str]): The API key for authentication.
        timeout (int): The default timeout in seconds for API requests.
        max_retries (int): The maximum number of retries for a failed request.
        logger (logging.Logger): A logger instance for logging API interactions.
        history (List[Dict[str, Any]]): A list to record all successful interactions.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        """
        Initializes the LanguageModelAPI client.

        Args:
            model_name (str): The identifier for the language model to use.
            api_key (Optional[str]): The API key for the service. It's recommended
                                     to load this from a secure source.
            timeout (int): Default request timeout in seconds.
            max_retries (int): Default maximum number of retries.
            logger (Optional[logging.Logger]): A pre-configured logger instance.
                                               If None, a default one is created.
            **kwargs: Additional keyword arguments for the specific backend.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None # Placeholder for a potential persistent session
        self.history: List[Dict[str, Any]] = []

        # Set up logging
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Configure a default handler if none exists to ensure logs are visible.
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"Initialized client for model: {self.model_name}")

    @abstractmethod
    async def _call_api(self, prompt: Any, **kwargs: Any) -> Any:
        """
        Makes a single, raw API call to the specific LM backend.
        """
        pass

    @abstractmethod
    def _validate_and_parse_response(self, response: Any, **kwargs) -> str:
        """
        Validates the raw API response and extracts the generated text.
        """
        pass

    def _format_prompt(self, prompt: Any) -> Any:
        """
        Formats the prompt into the required structure for the backend.
        Base implementation handles strings. Subclasses should override for
        more complex formats like chat history.
        """
        if isinstance(prompt, str):
            return prompt
        # If not a string, return as-is. Subclass should handle it.
        self.logger.debug(f"Prompt type is {type(prompt)}, passing as-is. Subclass should override _format_prompt if needed.")
        return prompt

    async def run(self, prompt: Any, batch_metrics: Optional[BatchMetrics] = None, **kwargs: Any) -> str:
        """
        Executes a single prompt, logs, records, and returns the response with timeout.
        """
        # Create a log-friendly version of the prompt
        if isinstance(prompt, str):
            log_prompt = prompt
        elif isinstance(prompt, list) and prompt:
            log_prompt = f"Conversation with {len(prompt)} messages, last: {prompt[-1]}"
        else:
            log_prompt = str(prompt)

        self.logger.debug(f"Running prompt: '{log_prompt[:100].strip()}...'")
        retries = kwargs.pop("max_retries", self.max_retries)
        timeout = kwargs.pop("timeout", self.timeout)
        prompt_index = kwargs.pop("prompt_index", -1)

        if batch_metrics:
            batch_metrics.increment_active()

        try:
            for attempt in range(retries):
                try:
                    # Format the prompt before calling the API
                    formatted_prompt = self._format_prompt(prompt)

                    # Wrap the API call with asyncio.wait_for to enforce a timeout.
                    raw_response = await asyncio.wait_for(
                        self._call_api(formatted_prompt, **kwargs),
                        timeout=timeout
                    )
                    processed_response = self._validate_and_parse_response(raw_response, **kwargs)

                    # Record the successful interaction in history
                    history_record = {
                        "prompt": prompt,
                        "response": processed_response,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.history.append(history_record)

                    # Record success metrics
                    if batch_metrics:
                        batch_metrics.add_success(prompt_index, attempt + 1)

                    self.logger.debug(f"Success. Response: '{processed_response[:100]}...'")
                    return processed_response

                except asyncio.TimeoutError as e:
                    error_type = "timeout"
                    error_msg = f"API call timed out after {timeout}s"
                    
                    if batch_metrics:
                        batch_metrics.add_error(error_type, error_msg, prompt_index, attempt + 1)
                    
                    self.logger.warning(f"{error_msg} on attempt {attempt + 1}.")
                    if attempt + 1 >= retries:
                        self.logger.critical("Max retries reached after timeout. Raising final exception.")
                        raise

                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Categorize errors
                    if "429" in error_str or "too many requests" in error_str:
                        error_type = "rate_limit"
                        wait_time = min(60 * (2 ** attempt), 300)
                    elif "503" in error_str or "service unavailable" in error_str:
                        error_type = "service_unavailable"
                        wait_time = min(30 * (2 ** attempt), 120)
                    elif "timeout" in error_str:
                        error_type = "timeout"
                        wait_time = min(10 * (2 ** attempt), 60)
                    elif "400" in error_str or "bad request" in error_str:
                        error_type = "bad_request"
                        wait_time = min(5 * (2 ** attempt), 30)
                    elif "auth" in error_str or "401" in error_str or "403" in error_str:
                        error_type = "authentication"
                        wait_time = min(5 * (2 ** attempt), 30)
                    else:
                        error_type = error_str
                        wait_time = min(1 * (4 ** attempt), timeout)
                    
                    if batch_metrics:
                        batch_metrics.add_error(error_type, str(e), prompt_index, attempt + 1)
                    
                    self.logger.debug(f"API call failed on attempt {attempt + 1} - {error_type}: {str(e)}")
                    
                    if attempt + 1 >= retries:
                        self.logger.critical(f"Max retries reached. Final error type: {error_type}")
                        raise

                    # Wait before retrying with categorized backoff
                    if attempt + 1 < retries:
                        self.logger.debug(f"Waiting {wait_time}s before retry {attempt + 2}")
                        await asyncio.sleep(wait_time)

        finally:
            if batch_metrics:
                batch_metrics.decrement_active()

        return ""  # Should not be reached

    async def run_batch(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Executes a batch of prompts concurrently, with optional rate limiting and comprehensive metrics.

        Args:
            prompts (List[str]): A list of prompts to execute.
            max_concurrent_calls (Optional[int]): The maximum number of API calls to run in parallel.
                                                   If None, all prompts are run concurrently.
            **kwargs: Backend-specific parameters to apply to all prompts in the batch.

        Returns:
            A list of generated texts, in the same order as the input prompts.
        """
        self.logger.info(f"Running a batch of {len(prompts)} prompts...")
        
        # Initialize metrics tracking
        metrics = BatchMetrics()
        max_concurrent_calls = kwargs.pop('max_concurrent_calls', None)
        print_summary_report = kwargs.pop('print_summary_report', True)
        staggering_delay = kwargs.pop('staggering_delay', 0.1)
        
        if max_concurrent_calls:
            semaphore = asyncio.Semaphore(max_concurrent_calls)
            self.logger.info(f"Concurrency limited to {max_concurrent_calls} parallel calls.")

            async def _semaphore_run(prompt: str, index: int) -> str:
                async with semaphore:
                    # Add small delay to space out requests
                    await asyncio.sleep((index % max_concurrent_calls) * staggering_delay)
                    return await self.run(prompt, batch_metrics=metrics, prompt_index=index, **kwargs)

            tasks = [
                asyncio.create_task(_run_and_return_index(i, _semaphore_run(prompt, i)))
                for i, prompt in enumerate(prompts)
            ]
        else:
            # Handle case without semaphore
            tasks = [
                asyncio.create_task(_run_and_return_index(i, self.run(
                    prompt, 
                    batch_metrics=metrics, 
                    prompt_index=i,
                    **{k: (v[i] if (isinstance(v, list) or isinstance(v, tuple)) else v) for k, v in kwargs.items()}
                )))
                for i, prompt in enumerate(prompts)
            ]

        # Create results placeholder
        results = [None] * len(prompts)

        # Process tasks with progress bar showing metrics
        with tqdm(total=len(tasks), desc="Processing batch") as pbar:
            for future in asyncio.as_completed(tasks):
                index, result_or_exception = await future
                results[index] = result_or_exception
                
                # Update progress bar with current metrics
                pbar.set_postfix(metrics.get_tqdm_postfix())
                pbar.update(1)

        # Process final results
        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.warning(f"Prompt {i+1} in batch failed: {str(res)}")
                final_results.append(f"Error: {res}")
            else:
                final_results.append(res)

        # Print comprehensive summary report
        if print_summary_report:
            metrics.print_summary_report(len(prompts), self.logger)
        
        return final_results

#@title `GenAIAPI` implementation
class GenAIAPI(LanguageModelAPI):
    """
    Google Generative AI implementation of the LanguageModelAPI.
    
    Supports both string prompts and chat-style conversation lists.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        """
        Initialize Google GenAI API client.
        
        Args:
            model_name (str): Model to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            api_key (Optional[str]): Google API key. If None, loads from GOOGLE_API_KEY env var
            timeout (int): Request timeout in seconds
            max_retries (int): Maximum retry attempts
            logger (Optional[logging.Logger]): Logger instance
            **kwargs: Additional config parameters (temperature, max_output_tokens, etc.)
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("API key must be provided or set in GOOGLE_API_KEY environment variable")
        
        super().__init__(model_name, api_key, timeout, max_retries, logger, **kwargs)
        
        # Initialize Google GenAI client
        self.client = genai.Client(api_key=self.api_key)
        
        # Store default config parameters
        self.default_config = {}
        
        self.logger.info(f"Initialized Google GenAI client with model: {self.model_name}")
    
    def _format_prompt(self, prompt: Any) -> str:
        """
        Format prompt for Google GenAI API.
        
        Args:
            prompt: Can be a string or list of chat messages
            
        Returns:
            Formatted prompt string
        """
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list):
            # Handle chat-style conversations
            # Convert list of messages to a single string
            formatted_parts = []
            for msg in prompt:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', str(msg))
                    formatted_parts.append(f"{role}: {content}")
                else:
                    formatted_parts.append(str(msg))
            return "\n".join(formatted_parts)
        else:
            self.logger.warning(f"Unexpected prompt type: {type(prompt)}, converting to string")
            return str(prompt)
    
    async def _call_api(self, prompt: str, **kwargs: Any) -> Any:
        """
        Make async API call to Google GenAI.
        
        Args:
            prompt (str): Formatted prompt string
            **kwargs: Additional parameters for this specific call
            
        Returns:
            Raw API response
        """
        # Merge default config with call-specific parameters
        config_params = {**self.default_config, **kwargs}
        
        # Remove non-config parameters that might be passed from parent class
        config_params.pop('max_retries', None)
        config_params.pop('timeout', None)
        
        # Create config object
        config = GenerateContentConfig(**config_params) if config_params else None
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response
        except Exception as e:
            self.logger.debug(f"Google GenAI API call failed: {str(e)}")
            raise
    
    def _validate_and_parse_response(self, response: Any, **kwargs) -> str:
        """
        Validate and extract text from Google GenAI response.
        
        Args:
            response: Raw API response object
            
        Returns:
            Generated text string
            
        Raises:
            ValueError: If response is invalid or empty
        """
        constraint = kwargs.get('response_schema', None)
        try:
            if not response:
                raise ValueError("Empty response from API")
            
            if not hasattr(response, 'text'):
                raise ValueError("Response missing 'text' attribute")
            
            text = response.text
            if not text or not text.strip():
                raise ValueError("Empty text in response")

            if constraint:
                is_valid, error = validate_genai_response_constraint(text, constraint)
                if not is_valid:
                    raise ValueError(f"Response does not conform to schema: {error}")
                
            return text.strip()
            
        except Exception as e:
            self.logger.debug(f"Failed to parse response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get usage information from the last response in history.
        
        Returns:
            Dictionary with usage metadata if available
        """
        if not self.history:
            return {}
        
        last_entry = self.history[-1]
        return last_entry.get('usage', {})
    
    def update_config(self, **kwargs: Any) -> None:
        """
        Update default configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        self.default_config.update(kwargs)
        self.logger.info(f"Updated config: {kwargs}")

#@title `VllmAPI` implementation
class VllmAPI(LanguageModelAPI):
    """
    vLLM implementation of the LanguageModelAPI.

    vLLM is a fast and memory-efficient inference and serving engine for LLMs.
    This implementation uses the OpenAI-compatible API interface that vLLM provides.

    Supports both string prompts and chat-style conversation lists.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = "EMPTY",
        timeout: int = 60,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        load_balance: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize vLLM API client.

        Args:
            model_name (str): Model to use (should match the model served by vLLM)
            base_url (str): Base URL of the vLLM server (default: "http://localhost:8000/v1")
                           Can be a comma-separated list of URLs for load balancing
            api_key (Optional[str]): API key. vLLM typically doesn't require authentication,
                                     so "EMPTY" is used by default
            timeout (int): Request timeout in seconds
            max_retries (int): Maximum retry attempts
            logger (Optional[logging.Logger]): Logger instance
            load_balance (bool): Enable round-robin load balancing across multiple servers
            **kwargs: Additional config parameters (temperature, max_tokens, top_p, etc.)
        """
        # Get base URL from environment if available
        base_url = os.getenv('VLLM_BASE_URL', base_url)

        super().__init__(model_name, api_key, timeout, max_retries, logger, **kwargs)

        # Handle multiple base URLs for load balancing
        self.base_urls = [url.strip() for url in base_url.split(',')]
        self.load_balance = load_balance or len(self.base_urls) > 1
        self.current_url_index = 0

        # Initialize OpenAI clients for each server
        if self.load_balance:
            self.clients = [
                AsyncOpenAI(base_url=url, api_key=self.api_key, timeout=self.timeout)
                for url in self.base_urls
            ]
            self.logger.info(f"Initialized vLLM client with load balancing across {len(self.base_urls)} servers")
            for i, url in enumerate(self.base_urls):
                self.logger.info(f"  Server {i}: {url}")
        else:
            self.client = AsyncOpenAI(
                base_url=self.base_urls[0],
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self.logger.info(f"Initialized vLLM client with model: {self.model_name} at {self.base_urls[0]}")

        # Store default config parameters
        self.default_config = {}

    def _format_prompt(self, prompt: Any) -> List[Dict[str, str]]:
        """
        Format prompt for vLLM API (OpenAI-compatible format).

        Args:
            prompt: Can be a string or list of chat messages

        Returns:
            List of message dictionaries in OpenAI chat format
        """
        if isinstance(prompt, str):
            # Convert simple string to chat format
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # Handle chat-style conversations
            formatted_messages = []
            for msg in prompt:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Already in correct format
                    formatted_messages.append(msg)
                elif isinstance(msg, dict):
                    # Has dict structure but might need reformatting
                    role = msg.get('role', 'user')
                    content = msg.get('content', str(msg))
                    formatted_messages.append({"role": role, "content": content})
                else:
                    # Convert other types to user messages
                    formatted_messages.append({"role": "user", "content": str(msg)})
            return formatted_messages
        else:
            self.logger.warning(f"Unexpected prompt type: {type(prompt)}, converting to string")
            return [{"role": "user", "content": str(prompt)}]

    def _get_client(self) -> AsyncOpenAI:
        """
        Get the appropriate client for the next request.
        Uses round-robin load balancing if multiple servers are configured.

        Returns:
            AsyncOpenAI client instance
        """
        if not self.load_balance:
            return self.client

        # Round-robin load balancing
        client = self.clients[self.current_url_index]
        self.current_url_index = (self.current_url_index + 1) % len(self.clients)
        return client

    async def _call_api(self, prompt: List[Dict[str, str]], **kwargs: Any) -> Any:
        """
        Make async API call to vLLM server.

        Args:
            prompt (List[Dict[str, str]]): Formatted prompt as list of message dicts
            **kwargs: Additional parameters for this specific call

        Returns:
            Raw API response
        """
        # Merge default config with call-specific parameters
        config_params = {**self.default_config, **kwargs}

        # Remove non-API parameters that might be passed from parent class
        config_params.pop('max_retries', None)
        config_params.pop('timeout', None)
        config_params.pop('prompt_index', None)

        # Handle response_schema for constrained decoding
        response_schema = config_params.pop('response_schema', None)
        guided_decoding_backend = config_params.pop('guided_decoding_backend', None)

        if response_schema:
            # Convert to vLLM's guided_json format
            # vLLM expects the schema as a JSON object
            import json

            # Prepare extra_body for guided generation
            extra_body = config_params.get('extra_body', {})

            if isinstance(response_schema, dict):
                extra_body['guided_json'] = response_schema
            elif isinstance(response_schema, str):
                extra_body['guided_json'] = json.loads(response_schema)

            # Optionally specify the backend (outlines, lm-format-enforcer, or xgrammar)
            # If not specified, vLLM will use its default
            if guided_decoding_backend:
                extra_body['guided_decoding_backend'] = guided_decoding_backend

            config_params['extra_body'] = extra_body
            self.logger.debug(f"Using guided JSON generation with schema")

        try:
            # Get client (handles load balancing if enabled)
            client = self._get_client()

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                **config_params
            )
            return response
        except Exception as e:
            self.logger.debug(f"vLLM API call failed: {str(e)}")
            raise

    def _validate_and_parse_response(self, response: Any, **kwargs) -> str:
        """
        Validate and extract text from vLLM response.

        Args:
            response: Raw API response object (OpenAI-compatible format)
            **kwargs: Additional validation parameters

        Returns:
            Generated text string

        Raises:
            ValueError: If response is invalid or empty
        """
        try:
            if not response:
                raise ValueError("Empty response from API")

            if not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Response missing 'choices' attribute or choices are empty")

            # Extract text from first choice
            choice = response.choices[0]
            if not hasattr(choice, 'message'):
                raise ValueError("Choice missing 'message' attribute")

            message = choice.message
            if not hasattr(message, 'content'):
                raise ValueError("Message missing 'content' attribute")

            text = message.content
            if not text or not text.strip():
                raise ValueError("Empty text in response")

            # Optional: validate against response schema if provided
            constraint = kwargs.get('response_schema', None)
            if constraint:
                is_valid, error = validate_genai_response_constraint(text, constraint)
                if not is_valid:
                    raise ValueError(f"Response does not conform to schema: {error}")

            return text.strip()

        except Exception as e:
            self.logger.debug(f"Failed to parse response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get usage information from the last response in history.

        Returns:
            Dictionary with usage metadata if available
        """
        if not self.history:
            return {}

        last_entry = self.history[-1]
        return last_entry.get('usage', {})

    def update_config(self, **kwargs: Any) -> None:
        """
        Update default configuration parameters.

        Common parameters for vLLM:
        - temperature: Float (0.0 to 2.0)
        - max_tokens: Integer
        - top_p: Float (0.0 to 1.0)
        - top_k: Integer
        - frequency_penalty: Float (-2.0 to 2.0)
        - presence_penalty: Float (-2.0 to 2.0)
        - stop: List[str] or str

        Args:
            **kwargs: Configuration parameters to update
        """
        self.default_config.update(kwargs)
        self.logger.info(f"Updated config: {kwargs}")