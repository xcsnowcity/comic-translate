from typing import Any
import numpy as np
import requests
import json

from .base import BaseLLMTranslation
from ...utils.translator_utils import MODEL_MAP
from ...utils.textblock import TextBlock


class GeminiTranslation(BaseLLMTranslation):
    """Translation engine using Google Gemini models via REST API."""
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.api_key = None
        self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize Gemini translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            model_name: Gemini model name
        """
        super().initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        credentials = settings.get_credentials(settings.ui.tr('Google Gemini'))
        self.api_key = credentials.get('api_key', '')
        
        # Map friendly model name to API model name
        self.model = MODEL_MAP.get(self.model_name)
        self.source_lang_en = source_lang
        self.target_lang_en = target_lang
    
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using Gemini REST API.
        
        Args:
            user_prompt: The prompt to send to the model
            system_prompt: System instructions for the model
            image: Image data as numpy array
            
        Returns:
            Translated text from the model
        """
        # Create API endpoint URL
        url = f"{self.api_base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Setup generation config
        generation_config = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
            "topP": self.top_p,
            "thinkingConfig": {
                "thinkingBudget": 0
            },
        }
        
        # Setup safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Prepare parts for the request
        parts = []
        
        # Add image if needed
        if self.img_as_llm_input:
            # Base64 encode the image

            img_b64, mime_type = self.encode_image(image)
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": img_b64
                }
            })
        
        # Add text prompt
        parts.append({"text": user_prompt})
        
        # Create the request payload
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }
        
        # Add system instructions if provided
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        
        # Send request to Gemini API
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            url, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        # Handle response
        if response.status_code != 200:
            error_msg = f"API request failed with status code {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        # Extract text from response
        response_data = response.json()
        
        try:
            # Extract the generated text from the response
            candidates = response_data.get("candidates", [])
            if not candidates:
                return "No response generated"
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            # Concatenate all text parts
            result = ""
            for part in parts:
                if "text" in part:
                    result += part["text"]
            
            return result
        except (KeyError, IndexError) as e:
            raise Exception(f"Failed to parse API response: {str(e)}")

    def translate_image_to_textblocks(self, image: np.ndarray) -> list[TextBlock]:
        # Construct a prompt that asks the LLM to identify and translate text in the image
        user_prompt = f"Identify all text in the image. For each piece of text, provide the original text and its {self.target_lang_en} translation. Format the output as a JSON array of objects, where each object has 'original_text' and 'translated_text' keys."
        
        system_prompt = self.get_system_prompt(self.source_lang_en, self.target_lang_en)
        
        # Set img_as_llm_input to True to send the image to the LLM
        self.img_as_llm_input = True
        
        response_text = self._perform_translation(user_prompt, system_prompt, image)
        
        # Parse the JSON response from the LLM
        try:
            # Remove markdown code block if present
            if response_text.startswith('```json') and response_text.endswith('```'):
                response_text = response_text[len('```json'):-len('```')].strip()
            parsed_response = json.loads(response_text)
            if not isinstance(parsed_response, list):
                raise ValueError("LLM response is not a JSON array.")
            
            text_blocks = []
            for item in parsed_response:
                original_text = item.get("original_text", "")
                translated_text = item.get("translated_text", "")
                
                # Create TextBlock objects. Since we don't have precise bounding boxes
                # from this LLM call, we'll initialize with dummy coordinates or None.
                # The TextBlock constructor might need to be flexible for this.
                # For now, assuming TextBlock can be created with just text.
                # If the LLM provides coordinates, they can be parsed and used here.
                blk = TextBlock(text=original_text, translation=translated_text)
                text_blocks.append(blk)
            return text_blocks
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse LLM response as JSON: {e}. Response: {response_text}")
        except Exception as e:
            raise Exception(f"Error processing LLM response: {e}. Response: {response_text}")