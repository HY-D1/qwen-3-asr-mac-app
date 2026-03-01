#!/usr/bin/env python3
"""
Simple LLM Text Processor - Multiple Backend Options
Just works™ - picks the first available backend

Improved prompts with detailed instructions, examples, and post-processing.
"""

import os
import sys
import json
import re
import traceback
from typing import Optional, Dict, Any, List


# =============================================================================
# Context Detection and Prompt Builder
# =============================================================================

class ContextDetector:
    """Detect context type from text to optimize prompts"""
    
    # Keywords for different context types
    MEETING_KEYWORDS = [
        'meeting', 'discuss', 'agenda', 'action item', 'follow up', 'team',
        'stakeholder', 'deadline', 'milestone', 'project', 'sync', 'standup',
        'review', 'quarterly', 'weekly', 'monthly', 'schedule', 'calendar',
        'attendee', 'participant', 'chair', 'minutes', 'vote', 'decision'
    ]
    
    LECTURE_KEYWORDS = [
        'lecture', 'professor', 'student', 'class', 'course', 'university',
        'academic', 'study', 'research', 'theory', 'concept', 'principle',
        'example', 'homework', 'assignment', 'exam', 'grade', 'semester',
        'curriculum', 'discipline', 'field', 'subject', 'topic', 'lesson'
    ]
    
    CONVERSATION_KEYWORDS = [
        'chat', 'talk', 'conversation', 'dialogue', 'interview', 'call',
        'phone', 'video', 'zoom', 'hello', 'hi', 'hey', 'thanks', 'please',
        'sorry', 'maybe', 'think', 'feel', 'believe', 'opinion', 'guess'
    ]
    
    PRESENTATION_KEYWORDS = [
        'presentation', 'slide', 'deck', 'demo', 'showcase', 'webinar',
        'conference', 'talk', 'keynote', 'workshop', 'panel', 'session',
        'q&a', 'question', 'answer', 'audience', 'speaker', 'host'
    ]
    
    @classmethod
    def detect(cls, text: str) -> str:
        """
        Detect the context type of the text.
        Returns: 'meeting', 'lecture', 'presentation', 'conversation', or 'general'
        """
        text_lower = text.lower()
        scores = {
            'meeting': sum(1 for kw in cls.MEETING_KEYWORDS if kw in text_lower),
            'lecture': sum(1 for kw in cls.LECTURE_KEYWORDS if kw in text_lower),
            'presentation': sum(1 for kw in cls.PRESENTATION_KEYWORDS if kw in text_lower),
            'conversation': sum(1 for kw in cls.CONVERSATION_KEYWORDS if kw in text_lower),
        }
        
        # Return the highest scoring context, or general if all scores are low
        max_score = max(scores.values())
        if max_score < 2:
            return 'general'
        return max(scores, key=scores.get)


class PromptBuilder:
    """Build optimized prompts with examples and context awareness"""
    
    @staticmethod
    def get_system_prompt(mode: str, context: str = "general") -> str:
        """Get the appropriate system prompt with context awareness"""
        
        base_prompts = {
            "punctuate": PromptBuilder._punctuate_prompt(context),
            "summarize": PromptBuilder._summarize_prompt(context),
            "clean": PromptBuilder._clean_prompt(context),
            "key_points": PromptBuilder._key_points_prompt(context),
            "format": PromptBuilder._format_prompt(context),
            "paragraph": PromptBuilder._paragraph_prompt(context),
        }
        
        return base_prompts.get(mode, base_prompts["punctuate"])
    
    @staticmethod
    def _punctuate_prompt(context: str) -> str:
        """Enhanced punctuate prompt with examples and context"""
        
        context_specific = {
            "meeting": "Focus on clarity for meeting minutes. Ensure action items and decisions are clearly punctuated.",
            "lecture": "Preserve academic terminology. Ensure technical terms are correctly capitalized.",
            "presentation": "Format for readability. Use punctuation that emphasizes key points.",
            "conversation": "Keep a natural, conversational flow while adding clarity.",
            "general": "Maintain the original tone and style while improving readability."
        }
        
        return f"""You are an expert transcription editor specializing in ASR (Automatic Speech Recognition) output refinement.

TASK: Add proper punctuation, capitalization, and formatting to the provided text while preserving the exact original meaning.

{context_specific.get(context, context_specific["general"])}

DETAILED GUIDELINES:

1. SENTENCE BOUNDARIES
   - Add periods (.) at natural sentence endings
   - Break long run-on sentences into shorter, clear sentences
   - Use question marks (?) for questions and exclamation marks (!) for emphasis

2. CAPITALIZATION
   - Capitalize the first word of every sentence
   - Capitalize proper nouns (names, places, companies, products)
   - Capitalize titles and honorifics (Mr., Mrs., Dr., Prof., CEO, President)
   - Capitalize days of the week, months, holidays
   - Capitalize acronyms and initialisms (NASA, FBI, CEO)

3. PUNCTUATION
   - Add commas for natural pauses and to separate clauses
   - Use commas in lists (Oxford comma optional but be consistent)
   - Add apostrophes for contractions and possessives
   - Use hyphens for compound modifiers and number ranges
   - Use em-dashes (—) or colons (:) for emphasis or elaboration

4. ASR ERROR CORRECTION
   - Fix obvious homophone errors based on context (their/there/they're, your/you're)
   - Correct number formatting ("three pm" → "3:00 PM")
   - Fix common ASR artifacts and repeated words

5. PRESERVATION RULES
   - DO NOT change the meaning or content of the text
   - DO NOT remove information, even if it seems redundant
   - DO NOT rewrite sentences - only add punctuation and capitalization
   - Preserve all technical terms, names, and specific details exactly

EXAMPLES:

Example 1 - Business Meeting:
Input: "so john from marketing said we need to launch by friday uh thats march fifteenth and sarah agreed but tom said we need more time for testing"
Output: "So John from Marketing said we need to launch by Friday, uh, that's March 15th, and Sarah agreed. But Tom said we need more time for testing."

Example 2 - Academic Lecture:
Input: "the theory of relativity developed by einstein in nineteen oh five changed our understanding of space and time basically it says that the laws of physics are the same for all non-accelerating observers"
Output: "The Theory of Relativity, developed by Einstein in 1905, changed our understanding of space and time. Basically, it says that the laws of physics are the same for all non-accelerating observers."

Example 3 - Technical Discussion:
Input: "we need to implement the api using restful principles so get post put and delete operations should be idempotent meaning multiple identical requests should have the same effect as a single request"
Output: "We need to implement the API using RESTful principles. So GET, POST, PUT, and DELETE operations should be idempotent, meaning multiple identical requests should have the same effect as a single request."

CRITICAL: Return ONLY the punctuated text. No explanations, no "Here is the result:", no quotes around the output."""

    @staticmethod
    def _summarize_prompt(context: str) -> str:
        """Enhanced summarize prompt with examples and context"""
        
        context_specific = {
            "meeting": "Focus on decisions made, action items assigned, and key outcomes. Include who is responsible for what.",
            "lecture": "Capture the main concepts, theories, and key takeaways. Preserve important definitions and examples.",
            "presentation": "Highlight the main message, key findings, and conclusions. Include important data points mentioned.",
            "conversation": "Summarize the main topics discussed and any agreements or conclusions reached.",
            "general": "Extract the most important information while maintaining the core message."
        }
        
        return f"""You are an expert summarizer specializing in condensing ASR transcripts while preserving key information.

TASK: Create a concise, informative summary of the provided text.

{context_specific.get(context, context_specific["general"])}

DETAILED GUIDELINES:

1. CONTENT SELECTION
   - Extract main topics and central ideas
   - Include key facts, figures, and specific details that matter
   - Identify and preserve important conclusions or decisions
   - Maintain chronological order if events are described sequentially

2. CONCISENESS
   - Remove redundant or repetitive information
   - Eliminate off-topic tangents and digressions
   - Consolidate similar points into single statements
   - Aim for approximately 20-30% of the original length

3. CLARITY
   - Use clear, direct language
   - Write in complete sentences
   - Ensure the summary can stand alone without the original text
   - Maintain logical flow and coherence

4. PRESERVATION
   - Keep important names, dates, numbers, and specifics
   - Preserve the original intent and tone
   - Don't add information not present in the original
   - Don't include your own opinions or analysis

EXAMPLES:

Example 1 - Business Meeting:
Input: "Okay so welcome everyone to our weekly sync um today we need to discuss the Q3 roadmap first john can you give us an update on the mobile app john says were on track for the beta release next tuesday but we need to fix the login bug sarah mentioned shes working on the api integration and should be done by friday tom raised concerns about the server costs he thinks we need to optimize the database queries before launch we agreed to prioritize performance optimization for next sprint action items john will fix login bug by monday sarah will complete api integration by friday tom will analyze server costs and propose optimizations okay great meeting everyone"
Output: "Weekly sync covered Q3 roadmap updates. John confirmed mobile app beta is on track for Tuesday, pending login bug fix. Sarah is completing API integration by Friday. Tom raised server cost concerns; team agreed to prioritize performance optimization next sprint. Action items: John (login bug fix by Monday), Sarah (API by Friday), Tom (server cost analysis)."

Example 2 - Academic Lecture:
Input: "today were discussing photosynthesis so plants convert light energy into chemical energy this process occurs in the chloroplasts specifically in the thylakoid membranes there are two stages the light dependent reactions and the calvin cycle in the light reactions water is split producing oxygen and atp and nadph are generated the calvin cycle uses these to convert co2 into glucose key factors affecting photosynthesis include light intensity carbon dioxide concentration and temperature understanding photosynthesis is crucial for agriculture and addressing climate change"
Output: "Photosynthesis converts light energy into chemical energy in plant chloroplasts. The process has two stages: light-dependent reactions (occurring in thylakoid membranes, producing oxygen, ATP, and NADPH) and the Calvin cycle (converting CO2 into glucose using ATP and NADPH). Key factors affecting the process are light intensity, CO2 concentration, and temperature. Photosynthesis knowledge is essential for agriculture and climate change research."

Example 3 - Product Discussion:
Input: "um so customer feedback shows that users really love the new dashboard feature they say its much easier to track metrics however some people are confused about the export function they cant find the csv download button also mobile usage has increased forty percent since last quarter this suggests we should prioritize mobile optimization for the next release the top requested features are dark mode integrations with slack and automated reporting"
Output: "Customer feedback indicates strong satisfaction with the new dashboard feature, particularly for metric tracking. Main pain point: users struggle to locate the CSV export function. Mobile usage has grown 40% since last quarter, indicating need for mobile optimization priority. Top feature requests: dark mode, Slack integration, and automated reporting."

CRITICAL: Return ONLY the summary text. No bullet points unless specifically requested, no "Summary:", no explanatory text."""

    @staticmethod
    def _clean_prompt(context: str) -> str:
        """Enhanced clean prompt with examples and context"""
        
        context_specific = {
            "meeting": "Focus on making meeting minutes professional and concise. Remove hesitations while keeping the business context clear.",
            "lecture": "Clean up speech disfluencies but preserve the academic tone and technical language exactly as spoken.",
            "presentation": "Create polished, professional text suitable for documentation or sharing. Remove casual speech patterns.",
            "conversation": "Keep the conversational tone natural while removing obvious verbal fillers and false starts.",
            "general": "Transform spoken text into polished written text while preserving meaning and tone."
        }
        
        return f"""You are an expert speech editor specializing in removing disfluencies and polishing ASR transcripts.

TASK: Remove filler words, fix grammar, and clean up the text while preserving the original meaning.

{context_specific.get(context, context_specific["general"])}

DETAILED GUIDELINES:

1. FILLER WORDS TO REMOVE
   - Common fillers: um, uh, like, you know, sort of, kind of
   - Hesitation markers: well, so, actually, basically, literally
   - Redundant phrases: "I mean", "you see", "the thing is"
   - Speech tics: repeated "and", "so", "but" at sentence starts

2. FALSE STARTS AND SELF-CORRECTIONS
   - Remove incomplete sentences that are immediately corrected
   - Keep the corrected version only
   - Example: "we need to uh no we should focus on" → "we should focus on"
   - Example: "the budget is is around fifty thousand" → "the budget is around fifty thousand"

3. REPETITIONS
   - Remove repeated words ("the the", "and and")
   - Remove repeated phrases ("we need to we need to")
   - Keep intentional repetition for emphasis

4. GRAMMAR FIXES
   - Fix subject-verb agreement errors
   - Correct obvious tense inconsistencies
   - Fix article usage (a/an/the)
   - Improve sentence structure while keeping original wording
   - Don't over-correct casual but acceptable speech patterns

5. PRESERVATION
   - Keep all substantive content and information
   - Preserve technical terms and jargon
   - Maintain the speaker's voice and tone
   - Don't change the meaning of statements

EXAMPLES:

Example 1 - Casual Conversation:
Input: "um so like i was thinking we could maybe um you know meet tomorrow at like three pm or something uh if that works for you basically i just want to discuss the project timeline you know what i mean"
Output: "I was thinking we could meet tomorrow at 3 PM if that works for you. I just want to discuss the project timeline."

Example 2 - Business Meeting:
Input: "uh okay so john you know will present the q3 results um actually no sarah will do that and then well sort of discuss the budget basically the main thing is we need to cut costs by ten percent you know what im saying"
Output: "John will present the Q3 results, then Sarah will discuss the budget. The main thing is we need to cut costs by 10%."

Example 3 - Technical Discussion:
Input: "the api endpoint um returns a json response with uh status code two hundred when successful and like four hundred four when not found basically you need to handle both cases in your code sort of wrap it in a try catch block you know"
Output: "The API endpoint returns a JSON response with status code 200 when successful and 404 when not found. You need to handle both cases in your code, wrapping it in a try-catch block."

Example 4 - Lecture:
Input: "so quantum mechanics um fundamentally challenges our classical intuition you know the wave particle duality basically means that particles exhibit both wave and particle properties uh depending on how we observe them this is uh you know one of the most profound insights in modern physics"
Output: "Quantum mechanics fundamentally challenges our classical intuition. The wave-particle duality means that particles exhibit both wave and particle properties depending on how we observe them. This is one of the most profound insights in modern physics."

CRITICAL: Return ONLY the cleaned text. No explanations, no "Cleaned text:", no notes about what was removed."""

    @staticmethod
    def _key_points_prompt(context: str) -> str:
        """Enhanced key points prompt with examples and context"""
        
        context_specific = {
            "meeting": "Focus on action items, decisions made, deadlines, and who is responsible for what. Include any concerns raised.",
            "lecture": "Extract key concepts, definitions, important facts, and takeaways. Preserve specific examples that illustrate concepts.",
            "presentation": "Capture main findings, recommendations, data points, and conclusions. Include any Q&A highlights.",
            "conversation": "Identify main topics, agreements, concerns, and any next steps or follow-ups mentioned.",
            "general": "Extract the most important and actionable information from the text."
        }
        
        return f"""You are an expert at extracting key information from ASR transcripts and organizing it into clear, actionable bullet points.

TASK: Extract key points, facts, decisions, and action items from the provided text.

{context_specific.get(context, context_specific["general"])}

DETAILED GUIDELINES:

1. CONTENT SELECTION
   - Extract specific facts, figures, dates, and numbers
   - Identify decisions that were made
   - Capture action items with owners and deadlines
   - Note important concerns, risks, or issues raised
   - Include key recommendations or conclusions

2. ORGANIZATION
   - Group related points together logically
   - Prioritize by importance (most critical first)
   - Use sub-bullets for related details
   - Keep bullets concise but informative

3. FORMATTING
   - Use "•" (bullet point) as the main bullet character
   - Start each bullet with a capital letter
   - End bullets with periods for complete sentences
   - Use bold for emphasis on critical items when appropriate
   - Create sections with headers if multiple distinct topics exist

4. CLARITY
   - Make each bullet self-contained and understandable
   - Include context needed to understand the point
   - Be specific rather than vague (names, dates, numbers)
   - Avoid generic statements without supporting details

EXAMPLES:

Example 1 - Business Meeting:
Input: "welcome to the project kickoff so first we need to establish the timeline john will handle the design phase starting monday he needs two weeks for that then sarah takes over for development which should take about three weeks we also discussed budget constraints the initial estimate was fifty thousand but finance approved only forty five so we need to cut some features tom suggested removing the advanced analytics module for now and adding it in phase two everyone agreed thats a good compromise one concern raised was the api dependency on the vendor they havent confirmed the delivery date yet action items john sends wireframes by friday sarah reviews api docs over the weekend tom schedules vendor call for monday next meeting is next wednesday"
Output: "• **Project Timeline Established**: Design phase (John, 2 weeks starting Monday) → Development phase (Sarah, 3 weeks).
• **Budget Approved**: $45K (down from $50K estimate).
• **Scope Adjustment**: Advanced analytics module deferred to Phase 2 to meet budget.
• **Risk Identified**: Vendor API delivery date unconfirmed.

**Action Items:**
• John: Send wireframes by Friday.
• Sarah: Review API documentation over the weekend.
• Tom: Schedule vendor call for Monday.
• **Next Meeting**: Wednesday (date TBD)."

Example 2 - Academic Lecture:
Input: "today were covering the causes of world war one the main factors were militarism alliances imperialism and nationalism the assassination of archduke franz ferdinand in sarajevo on june twenty eighth nineteen fourteen was the immediate trigger austria hungary declared war on serbia on july twenty eighth this activated alliance systems russia mobilized to defend serbia germany backed austria hungary and declared war on russia august first then germany invaded belgium to attack france bringing britain into the war the schlieffen plan failed and the war became a stalemate with trench warfare dominating the western front key consequences included the collapse of four empires and redrawing of the european map"
Output: "• **MAIN CAUSES (The 4 Is)**: Militarism, Alliances, Imperialism, Nationalism.
• **Immediate Trigger**: Assassination of Archduke Franz Ferdinand in Sarajevo (June 28, 1914).
• **Escalation Timeline**:
  - July 28: Austria-Hungary declares war on Serbia.
  - Russia mobilizes to defend Serbia.
  - August 1: Germany declares war on Russia.
  - Germany invades Belgium → Britain enters war.
• **Military Developments**: Schlieffen Plan failed; Western Front becomes trench warfare stalemate.
• **Major Consequences**: Collapse of four empires; redrawing of European map."

Example 3 - Product Review:
Input: "customer feedback session results users love the new ui they say its much more intuitive especially the navigation menu the main complaints are about loading times on mobile average load time is four seconds which is too slow also the search function doesnt work well with typos users want autocomplete features and the ability to filter results integration requests keep coming up especially slack and salesforce people want automated workflows the pricing page is confusing many prospects dont understand the tier differences we need to simplify that support tickets have increased twenty percent since the last release mostly about the new export feature"
Output: "• **Positive Feedback**: New UI praised as intuitive, especially navigation menu.
• **Performance Issues**: Mobile load times averaging 4 seconds (needs optimization).
• **Search Problems**: Poor typo handling; users want autocomplete and filtering.
• **Integration Requests**: Slack and Salesforce integrations most requested; automated workflows desired.
• **Pricing Concerns**: Page confusing, tier differences unclear to prospects.
• **Support Trend**: Tickets up 20% since last release, mainly regarding new export feature."

CRITICAL: Return ONLY the bullet points. No introductory text like "Here are the key points:", no conclusion, no additional commentary."""

    @staticmethod
    def _format_prompt(context: str) -> str:
        """Enhanced format prompt with examples and context"""
        
        context_specific = {
            "meeting": "Format as structured meeting notes with clear sections for attendees, agenda items, decisions, and action items.",
            "lecture": "Format as lecture notes with clear topic headers, key concepts, definitions, and examples separated into sections.",
            "presentation": "Format as presentation summary with clear sections covering key slides, main points, and conclusions.",
            "conversation": "Format as discussion notes with clear topic headings and key exchanges organized logically.",
            "general": "Format as well-structured notes with clear sections and logical organization."
        }
        
        return f"""You are an expert at formatting unstructured ASR transcripts into well-organized, professional documents.

TASK: Format the provided text into structured notes with clear sections and hierarchy.

{context_specific.get(context, context_specific["general"])}

DETAILED GUIDELINES:

1. STRUCTURE
   - Use clear section headers (ALL CAPS or Title Case)
   - Organize content logically by topic or chronology
   - Use bullet points for lists and details
   - Include a brief overview/summary at the top if the text is long

2. SECTIONS TO INCLUDE (where applicable)
   - Overview/Summary
   - Key Topics/Agenda Items
   - Decisions Made
   - Action Items (with owners and deadlines)
   - Open Questions/Concerns
   - Next Steps

3. FORMATTING
   - Use consistent heading styles
   - Use bullet points (•) for details under sections
   - Use bold for emphasis on important items
   - Maintain professional appearance

4. CLARITY
   - Ensure each section is self-contained
   - Remove redundancy while keeping all unique information
   - Make the document scannable and easy to read

EXAMPLES:

Example 1 - Meeting:
Input: "attendees john sarah tom and mike agenda q3 review and q4 planning john presented sales numbers were up fifteen percent which beat our target sarah noted marketing costs were over budget by ten percent tom said engineering shipped all planned features mike raised concerns about hiring we need two more developers decisions we will increase marketing spend for q4 but only on digital channels the new product launch is postponed to january action items john will prepare q4 forecast by friday sarah will revise marketing budget by wednesday tom will post job openings by monday next meeting october fifteenth"
Output: "**MEETING NOTES**
*Attendees*: John, Sarah, Tom, Mike

**OVERVIEW**
Q3 review shows strong sales performance but marketing overspend. Q4 planning includes budget adjustments and hiring plans.

**Q3 PERFORMANCE**
• **Sales**: Up 15% (beat target) - John
• **Marketing**: 10% over budget - Sarah
• **Engineering**: All planned features shipped - Tom

**CONCERNS RAISED**
• Hiring: Need 2 additional developers - Mike

**DECISIONS MADE**
• Q4 marketing budget increased, but digital channels only.
• New product launch postponed to January.

**ACTION ITEMS**
• John: Prepare Q4 forecast by Friday.
• Sarah: Revise marketing budget by Wednesday.
• Tom: Post job openings by Monday.

**NEXT MEETING**: October 15th"

CRITICAL: Return ONLY the formatted notes. No "Here are the notes:" or other introductory text."""

    @staticmethod
    def _paragraph_prompt(context: str) -> str:
        """Enhanced paragraph formatting prompt"""
        
        return """You are an expert at structuring text into well-organized paragraphs.

TASK: Organize the text into logical paragraphs with proper flow and structure.

GUIDELINES:
1. Group related ideas into single paragraphs
2. Each paragraph should focus on one main idea or topic
3. Use appropriate transitions between paragraphs
4. Ensure smooth flow from one paragraph to the next
5. Keep paragraphs at a readable length (3-5 sentences ideally)
6. Add paragraph breaks at natural topic transitions

EXAMPLES:

Input: "first we need to discuss the budget the initial estimate was too low we should request more funding also the timeline needs adjustment the original deadline is not realistic next we need to review the team assignments everyone seems overloaded we might need to hire contractors finally lets talk about the deliverables the scope has crept and we need to refocus"

Output: "First, we need to discuss the budget. The initial estimate was too low, so we should request more funding. Additionally, the timeline needs adjustment because the original deadline is not realistic.

Next, we need to review the team assignments. Everyone seems overloaded, so we might need to hire contractors to help manage the workload.

Finally, let's talk about the deliverables. The scope has crept beyond what was originally planned, and we need to refocus on the core requirements."

CRITICAL: Return ONLY the paragraph-formatted text. No explanations or additional text."""


class PostProcessor:
    """Clean up and normalize LLM output"""
    
    # Common prefixes to remove from LLM responses
    PREFIXES_TO_REMOVE = [
        r"^(Here is|Here are|Below is|Below are)\s+(the|a|an)?\s*[^:]*:\s*",
        r"^(Punctuated text|Summary|Cleaned text|Key points|Formatted notes|Result)[:\-]?\s*",
        r"^(I've|The system|The assistant)\s+[^.]*\.\s*",
        r'^["\']',
        r'["\']$',
    ]
    
    # Suffixes to remove
    SUFFIXES_TO_REMOVE = [
        r"\n\n?(Note:|Disclaimer:|Important:|Please note)[^$]*$",
        r"\n\n?I hope this helps[^$]*$",
        r"\n\n?Let me know if[^$]*$",
    ]
    
    @classmethod
    def clean(cls, text: str, original_text: str, mode: str) -> str:
        """Clean up LLM output"""
        if not text or not text.strip():
            return original_text
        
        result = text.strip()
        
        # Remove common prefixes
        for pattern in cls.PREFIXES_TO_REMOVE:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE).strip()
        
        # Remove common suffixes
        for pattern in cls.SUFFIXES_TO_REMOVE:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE).strip()
        
        # Remove surrounding quotes if present
        if (result.startswith('"') and result.endswith('"')) or \
           (result.startswith("'") and result.endswith("'")):
            result = result[1:-1].strip()
        
        # Normalize whitespace
        result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
        result = re.sub(r'[ \t]+', ' ', result)  # Normalize spaces
        
        # For key_points mode, ensure proper bullet formatting
        if mode == "key_points":
            result = cls._normalize_bullets(result)
        
        # If result is too short or too different, return original
        if len(result) < len(original_text) * 0.3 and mode != "summarize":
            return original_text
        
        return result.strip()
    
    @classmethod
    def _normalize_bullets(cls, text: str) -> str:
        """Normalize bullet point formatting"""
        # Replace various bullet styles with standard bullet
        text = re.sub(r'^[\-\*•]\s*', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\d+[\.\)]\s+)', r'\1', text, flags=re.MULTILINE)
        return text


# =============================================================================
# Backend 0: Ollama (Local Open Source LLM - Best Free Option)
# =============================================================================

class OllamaBackend:
    """Use Ollama to run local open source models like Qwen, Llama, Phi"""
    
    def __init__(self, model: str = "qwen:1.8b"):
        self.model = model
        self.available = False
        self._init()
    
    def _init(self):
        try:
            import subprocess
            import json
            
            # Check if ollama is installed
            result = subprocess.run(
                ["which", "ollama"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return
            
            # Check if server is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.available = True
                print(f"✅ Using Ollama backend ({self.model})")
                
                # Pull model if not present
                if self.model not in result.stdout:
                    print(f"📥 Pulling {self.model}...")
                    subprocess.run(
                        ["ollama", "pull", self.model],
                        capture_output=True,
                        timeout=300
                    )
            else:
                # Try to start server
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                import time
                time.sleep(2)
                
                # Check again
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.available = True
                    print(f"✅ Ollama backend started ({self.model})")
                    
        except Exception as e:
            print(f"Ollama not available: {e}")
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if not self.available:
            return text
        
        import subprocess
        import json
        
        # Detect context for better prompting
        context = ContextDetector.detect(text)
        
        # Get enhanced system prompt
        system_prompt = PromptBuilder.get_system_prompt(mode, context)
        
        # Build user prompt
        user_prompt = f"Please process the following text:\n\n{text}"
        
        data = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": min(len(text) * 2, 4000)  # Dynamic token limit
            }
        }
        
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-X", "POST",
                    "http://localhost:11434/api/generate",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(data)
                ],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                output = response.get("response", text)
                # Post-process the output
                return PostProcessor.clean(output, text, mode)
        except Exception as e:
            print(f"Ollama processing error: {e}")
        
        return text


# =============================================================================
# Backend 1: OpenAI API (Easiest - just need API key)
# =============================================================================

class OpenAIBackend:
    """Use OpenAI API - most reliable, requires internet"""
    
    def __init__(self):
        self.client = None
        self.available = False
        self._init()
    
    def _init(self):
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.available = True
                print("✅ Using OpenAI API backend")
        except:
            pass
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if not self.available:
            return text
        
        # Detect context for better prompting
        context = ContextDetector.detect(text)
        
        # Get enhanced system prompt
        system_prompt = PromptBuilder.get_system_prompt(mode, context)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please process this text:\n\n{text}"}
                ],
                temperature=0.3,
                max_tokens=min(len(text.split()) * 3, 4000)
            )
            output = response.choices[0].message.content.strip()
            # Post-process the output
            return PostProcessor.clean(output, text, mode)
        except Exception as e:
            print(f"OpenAI error: {e}")
            return text


# =============================================================================
# Backend 2: Transformers Pipeline (Simple local model)
# =============================================================================

class TransformersBackend:
    """Use HuggingFace transformers - downloads model once
    
    Best models for Mac M1 8GB:
    - Qwen/Qwen2.5-0.5B-Instruct: ~1GB, very fast, fits easily
    - Qwen/Qwen2.5-1.5B-Instruct: ~3GB, better quality, still fits
    - mlx-community/Qwen2.5-0.5B-Instruct-4bit: Optimized for Apple Silicon
    """
    
    def __init__(self, model_size="0.5B"):
        self.pipeline = None
        self.available = False
        # Use models compatible with transformers 4.35.0
        # Qwen2.5 needs transformers 4.36+, so we use compatible alternatives
        # Note: These models will download on first use
        self.model_options = {
            "0.5B": "sshleifer/tiny-gpt2",             # Tiny model for testing
            "1.5B": "gpt2",                            # GPT-2 (not ideal but works)
            "3B": "gpt2-medium",                       # Larger GPT-2
        }
        self.model_name = self.model_options.get(model_size, self.model_options["0.5B"])
        self._init()
    
    def _init(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"📥 Loading {self.model_name} (first time may take 1-2 min)...")
            
            # Determine best device
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                dtype = torch.float16
                print("   Using Apple Metal (MPS) acceleration")
            elif torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
                print("   Using CUDA GPU")
            else:
                device = "cpu"
                dtype = torch.float32
                print("   Using CPU")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # For smaller models, don't use device_map to avoid issues
            if "0.5B" in self.model_name:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                self.model = self.model.to(device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.device = device
            self.available = True
            print(f"✅ Loaded {self.model_name}")
            
        except Exception as e:
            print(f"Transformers backend not available: {e}")
            import traceback
            traceback.print_exc()
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if not self.available:
            return text
        
        # Detect context for better prompting
        context = ContextDetector.detect(text)
        
        # Get enhanced system prompt
        system_prompt = PromptBuilder.get_system_prompt(mode, context)
        
        try:
            import torch
            
            # Build prompt with instruction format
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nPlease process this text:\n\n{text}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            
            # Move to appropriate device
            if hasattr(self, 'device') and self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            elif torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate with appropriate parameters for small models
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(len(text.split()) * 2, 500),  # Dynamic max tokens
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract assistant response
            if "<|im_start|>assistant" in result:
                result = result.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in result:
                result = result.split("<|im_end|>")[0]
            
            result = result.strip()
            
            # Post-process the output
            result = PostProcessor.clean(result, text, mode)
            
            # If empty or too short, return original
            if len(result) < len(text) * 0.5:
                return text
            
            return result
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return text


# =============================================================================
# Backend 3: Rule-based (Always works, no dependencies)
# =============================================================================

class RuleBasedBackend:
    """Simple rule-based text improvement - always available"""
    
    # Common abbreviations that need special handling
    ABBREVIATIONS = {
        'mr': 'Mr.',
        'mrs': 'Mrs.',
        'ms': 'Ms.',
        'dr': 'Dr.',
        'prof': 'Prof.',
        'st': 'St.',
        'ave': 'Ave.',
        'blvd': 'Blvd.',
        'rd': 'Rd.',
        'inc': 'Inc.',
        'corp': 'Corp.',
        'ltd': 'Ltd.',
        'llc': 'LLC',
        'jr': 'Jr.',
        'sr': 'Sr.',
        'vs': 'vs.',
        'etc': 'etc.',
        'eg': 'e.g.',
        'ie': 'i.e.',
        'am': 'AM',
        'pm': 'PM',
    }
    
    # Common filler words and phrases
    FILLERS = [
        'um', 'uh', 'like', 'you know', 'sort of', 'kind of',
        'actually', 'basically', 'literally', 'so', 'well',
        'i mean', 'you see', 'the thing is', 'right',
    ]
    
    # Sentence starter words (often repeated)
    SENTENCE_STARTERS = ['and', 'but', 'so', 'then', 'also', 'plus']
    
    def __init__(self):
        self.available = True
        print("✅ Using rule-based backend (basic improvements)")
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if mode == "punctuate":
            return self._punctuate(text)
        elif mode == "clean":
            return self._clean(text)
        elif mode == "summarize":
            return self._simple_summarize(text)
        elif mode == "key_points":
            return self._extract_key_points(text)
        elif mode == "format":
            return self._format(text)
        elif mode == "paragraph":
            return self._paragraph(text)
        else:
            return self._punctuate(text)
    
    def _punctuate(self, text: str) -> str:
        """Enhanced punctuation with better patterns"""
        import re
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common abbreviations (case insensitive)
        for abbr, replacement in self.ABBREVIATIONS.items():
            pattern = r'\b' + abbr + r'\b'
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Capitalize first letter of the text
        if text:
            text = text[0].upper() + text[1:]
        
        # Add periods after sentence-ending words if not present
        # Look for sentence boundaries (end of sentence markers followed by space and lowercase)
        text = re.sub(r'([.!?])(\s+)([a-z])', lambda m: m.group(1) + m.group(2) + m.group(3).upper(), text)
        
        # Capitalize after sentence-ending punctuation
        sentences = re.split(r'([.!?]\s+)', text)
        result = []
        for i, sent in enumerate(sentences):
            if i == 0:
                result.append(sent)
            elif sent and sent[0] in ' ":\'-' and len(sent) > 1:
                # Check if this segment follows a sentence ending
                if i > 0 and sentences[i-1] and sentences[i-1][-1] in '.!?' if len(sentences[i-1]) > 0 else False:
                    result.append(sent[0] + sent[1].upper() + sent[2:])
                else:
                    result.append(sent)
            elif sent and i > 0:
                # Check if previous ended with punctuation
                prev = sentences[i-1] if i > 0 else ''
                if prev and prev[-1] in '.!?' if len(prev) > 0 else False:
                    result.append(sent[0].upper() + sent[1:] if len(sent) > 1 else sent.upper())
                else:
                    result.append(sent)
            else:
                result.append(sent)
        
        text = ''.join(result)
        
        # Add periods for sentence boundaries based on structure
        # Look for sentence starters preceded by lowercase and space
        for starter in ['The', 'We', 'I', 'They', 'It', 'This', 'That', 'There', 'Here', 'But', 'So', 'However']:
            pattern = r'([a-z]) (\b' + starter + r'\b)'
            text = re.sub(pattern, r'\1. \2', text)
        
        # Add commas before conjunctions in lists (simple approach)
        text = re.sub(r'(\w+) (\w+) (\w+) and (\w+)', r'\1, \2, \3, and \4', text)
        
        # Fix common number formats
        text = re.sub(r'\b(\d{1,2}) (am|pm)\b', r'\1:00 \2', text, flags=re.IGNORECASE)
        
        # Add question marks for questions
        question_starters = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 
                            'do', 'does', 'did', 'can', 'could', 'would', 'will', 'shall']
        for starter in question_starters:
            pattern = r'\b' + starter + r'\b[^.!?]*$'
            if re.search(pattern, text, re.IGNORECASE) and not text.endswith('?'):
                # Only add ? if it looks like a question
                pass  # Skip for now to avoid false positives
        
        # Ensure text ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Capitalize I pronoun
        text = re.sub(r'\bi\b', 'I', text)
        
        # Capitalize common proper nouns (simple heuristics)
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']
        for day in days:
            text = re.sub(r'\b' + day + r'\b', day.capitalize(), text, flags=re.IGNORECASE)
        for month in months:
            text = re.sub(r'\b' + month + r'\b', month.capitalize(), text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _clean(self, text: str) -> str:
        """Enhanced filler word removal"""
        import re
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove filler words and phrases
        for filler in self.FILLERS:
            # Remove as whole word with optional surrounding spaces
            pattern = r'\s*\b' + re.escape(filler) + r'\b\s*'
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Remove repeated words (e.g., "the the", "and and")
        text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
        
        # Remove repeated phrases (simple 2-word pattern)
        text = re.sub(r'\b(\w+\s+\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
        
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Ensure ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text.strip()
    
    def _simple_summarize(self, text: str) -> str:
        """Enhanced summary extraction"""
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return text[:200] + "..." if len(text) > 200 else text
        
        # Score sentences by importance
        def score_sentence(sent):
            score = 0
            sent_lower = sent.lower()
            
            # Boost for keywords indicating importance
            important_words = ['important', 'key', 'main', 'critical', 'essential', 
                             'significant', 'major', 'primary', 'crucial', 'vital',
                             'conclusion', 'summary', 'result', 'outcome', 'decision']
            for word in important_words:
                if word in sent_lower:
                    score += 2
            
            # Boost for sentences with numbers or data
            if re.search(r'\d+', sent):
                score += 1
            
            # Boost for sentences starting with conclusion words
            conclusion_starters = ['in conclusion', 'to summarize', 'finally', 'overall',
                                  'therefore', 'thus', 'as a result', 'in summary']
            for starter in conclusion_starters:
                if sent_lower.startswith(starter):
                    score += 3
            
            # Slight boost for longer sentences (more content)
            word_count = len(sent.split())
            if 8 <= word_count <= 25:
                score += 1
            
            return score
        
        # Score and sort sentences
        scored = [(s, score_sentence(s)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Take top sentences (up to 3 or 20% of total)
        num_to_take = min(3, max(1, len(sentences) // 5))
        top_sentences = [s[0] for s in scored[:num_to_take]]
        
        # Restore original order
        summary = [s for s in sentences if s in top_sentences]
        
        return ' '.join(summary)
    
    def _extract_key_points(self, text: str) -> str:
        """Extract key points as bullets"""
        import re
        
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "• " + text[:200] if len(text) > 200 else "• " + text
        
        # Look for key sentences
        key_sentences = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            
            # Check for key indicators
            is_key = False
            
            # Contains action words
            action_words = ['need to', 'should', 'must', 'will', 'action', 'task',
                          'decided', 'agreed', 'approved', 'rejected', 'plan']
            for word in action_words:
                if word in sent_lower:
                    is_key = True
                    break
            
            # Contains numbers, dates, or specific data
            if re.search(r'\d+', sent) or re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w* \d+', sent_lower):
                is_key = True
            
            # Contains important markers
            important_markers = ['important', 'key', 'main', 'critical', 'note',
                               'remember', 'don\'t forget', 'priority']
            for marker in important_markers:
                if marker in sent_lower:
                    is_key = True
                    break
            
            if is_key:
                key_sentences.append(sent)
        
        # If no key sentences found, use all sentences
        if not key_sentences:
            key_sentences = sentences[:5]  # Limit to first 5
        
        # Format as bullets
        bullets = []
        for sent in key_sentences[:8]:  # Max 8 bullets
            # Clean up the sentence
            sent = sent.strip()
            if sent and not sent.endswith('.'):
                sent += '.'
            bullets.append(f"• {sent}")
        
        return '\n'.join(bullets)
    
    def _format(self, text: str) -> str:
        """Basic formatting into sections"""
        import re
        
        # First punctuate
        text = self._punctuate(text)
        
        # Split into paragraphs (by topic changes)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences into paragraphs (every 3-4 sentences)
        paragraphs = []
        current_para = []
        
        for i, sent in enumerate(sentences):
            current_para.append(sent)
            if len(current_para) >= 3 or i == len(sentences) - 1:
                paragraphs.append(' '.join(current_para))
                current_para = []
        
        if current_para:
            paragraphs.append(' '.join(current_para))
        
        return '\n\n'.join(paragraphs)
    
    def _paragraph(self, text: str) -> str:
        """Organize into paragraphs"""
        return self._format(text)


# =============================================================================
# Main Interface
# =============================================================================

class SimpleLLM:
    """
    Simple LLM interface that tries multiple backends
    Priority: Ollama > OpenAI > Transformers > Rule-based
    """
    
    def __init__(self, model_size="0.5B", prefer_openai=False, ollama_model="qwen:1.8b"):
        """
        Args:
            model_size: "0.5B", "1.5B", or "3B" (for local models)
            prefer_openai: If True, try OpenAI first even if it requires env var
            ollama_model: Ollama model to use (qwen:1.8b, llama3.2:3b, etc.)
        """
        self.backend = None
        self.backend_name = "none"
        self.model_size = model_size
        self.ollama_model = ollama_model
        
        # Try backends in order
        # 1. Try Ollama first (local, free, open source)
        self._try_ollama()
        
        # 2. Try OpenAI if key is set
        if not self.backend and (prefer_openai or os.getenv('OPENAI_API_KEY')):
            self._try_openai()
        
        # 3. Try local transformers
        if not self.backend:
            self._try_transformers(model_size)
        
        # 4. Fall back to rule-based
        if not self.backend:
            self._use_rule_based()
    
    def _try_ollama(self):
        """Try Ollama local LLM"""
        backend = OllamaBackend(self.ollama_model)
        if backend.available:
            self.backend = backend
            self.backend_name = f"ollama-{self.ollama_model}"
    
    def _try_openai(self):
        """Try OpenAI API"""
        backend = OpenAIBackend()
        if backend.available:
            self.backend = backend
            self.backend_name = "openai"
    
    def _try_transformers(self, model_size):
        """Try local transformers model"""
        backend = TransformersBackend(model_size)
        if backend.available:
            self.backend = backend
            self.backend_name = "transformers"
    
    def _use_rule_based(self):
        """Fall back to rule-based"""
        self.backend = RuleBasedBackend()
        self.backend_name = "rule-based"
    
    def is_available(self) -> bool:
        return self.backend is not None and self.backend_name != "none"
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        """
        Process text
        
        Modes:
        - punctuate: Add punctuation and capitalization
        - summarize: Create summary
        - key_points: Extract key points
        - format: Format as meeting notes
        - clean: Remove filler words
        - paragraph: Organize into paragraphs
        """
        if not text or not text.strip():
            return text
        
        if not self.backend:
            return text
        
        return self.backend.process(text, mode)
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Simple analysis"""
        import re
        
        words = text.split()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Detect context
        context = ContextDetector.detect(text)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "char_count": len(text),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "backend_used": self.backend_name,
            "detected_context": context
        }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing SimpleLLM with improved prompts...\n")
    
    llm = SimpleLLM()
    
    test_text = "um so like we need to discuss the project timeline uh first we have the design phase then development and finally testing and deployment"
    
    print(f"Backend: {llm.backend_name}")
    print(f"\nOriginal:\n{test_text}\n")
    
    print(f"Punctuated:\n{llm.process(test_text, 'punctuate')}\n")
    print(f"Cleaned:\n{llm.process(test_text, 'clean')}\n")
    print(f"Summary:\n{llm.process(test_text, 'summarize')}\n")
    
    # Test with more complex text
    meeting_text = """okay so welcome everyone to our weekly sync um today we need to discuss the Q3 roadmap 
    first john can you give us an update on the mobile app john says were on track for the beta release next 
    tuesday but we need to fix the login bug sarah mentioned shes working on the api integration and should 
    be done by friday tom raised concerns about the server costs he thinks we need to optimize the database 
    queries before launch we agreed to prioritize performance optimization for next sprint action items john 
    will fix login bug by monday sarah will complete api integration by friday tom will analyze server costs 
    and propose optimizations okay great meeting everyone"""
    
    print("\n" + "="*60)
    print("Meeting text test:")
    print(f"\nOriginal:\n{meeting_text}\n")
    print(f"Key Points:\n{llm.process(meeting_text, 'key_points')}\n")
    print(f"Summary:\n{llm.process(meeting_text, 'summarize')}\n")
    
    analysis = llm.analyze(meeting_text)
    print(f"Analysis: {analysis}")
