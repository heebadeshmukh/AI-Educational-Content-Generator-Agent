# Educational AI Agent - Comprehensive Implementation
# This agent creates personalized educational content using free APIs and Gemini

import os
import json
import requests
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

# Core dependencies
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from google.colab import userdata
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    QUIZ = "quiz"
    FLASHCARD = "flashcard"
    SUMMARY = "summary"
    AUDIO = "audio"

@dataclass
class LearningMaterial:
    title: str
    content: str
    source: str
    difficulty_level: str
    subject: str
    key_concepts: List[str]

@dataclass
class GeneratedContent:
    content_type: ContentType
    title: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime

class APIManager:
    """Manages all external API integrations using free services"""
    
    def __init__(self):
        self.apis = {
            'gemini_key': os.getenv('GEMINI_API_KEY'),
            'youtube_key': os.getenv('YOUTUBE_API_KEY'),
            'openweather_key': os.getenv('OPENWEATHER_API_KEY')  # For demo purposes
        }
        
        # Initialize Gemini
        if self.apis['gemini_key']:
            genai.configure(api_key=self.apis['gemini_key'])
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
    def get_wikipedia_content(self, topic: str, language: str = 'en') -> Dict[str, Any]:
        """Free Wikipedia API for educational content"""
        try:
            # Search for the topic
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
            response = requests.get(search_url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', ''),
                    'extract': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'source': 'Wikipedia'
                }
        except Exception as e:
            logger.error(f"Wikipedia API error: {e}")
            
        return {}
    
    def get_youtube_educational_videos(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """YouTube Data API v3 (free tier: 100 requests/day)"""
        if not self.apis['youtube_key']:
            return []
            
        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                'part': 'snippet',
                'q': f"{query} educational tutorial",
                'type': 'video',
                'videoDuration': 'medium',
                'videoDefinition': 'high',
                'maxResults': max_results,
                'key': self.apis['youtube_key']
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                videos = []
                
                for item in data.get('items', []):
                    video = {
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'video_id': item['id']['videoId'],
                        'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'thumbnail': item['snippet']['thumbnails']['default']['url'],
                        'channel': item['snippet']['channelTitle']
                    }
                    videos.append(video)
                
                return videos
                
        except Exception as e:
            logger.error(f"YouTube API error: {e}")
            
        return []
    
    def get_open_library_books(self, subject: str) -> List[Dict[str, Any]]:
        """Open Library API - free book recommendations"""
        try:
            url = f"https://openlibrary.org/subjects/{subject}.json"
            params = {'limit': 10}
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                books = []
                
                for work in data.get('works', []):
                    book = {
                        'title': work.get('title', ''),
                        'authors': [author.get('name', '') for author in work.get('authors', [])],
                        'key': work.get('key', ''),
                        'url': f"https://openlibrary.org{work.get('key', '')}",
                        'subject': subject
                    }
                    books.append(book)
                
                return books
                
        except Exception as e:
            logger.error(f"Open Library API error: {e}")
            
        return []
    
    def get_free_course_content(self, topic: str) -> List[Dict[str, Any]]:
        """Simulate free course API (like edX, Coursera public content)"""
        # This would integrate with actual free course APIs
        # For demo, returning structured mock data
        courses = [
            {
                'title': f"Introduction to {topic}",
                'provider': 'MIT OpenCourseWare',
                'url': f"https://ocw.mit.edu/search/?q={topic}",
                'difficulty': 'Beginner',
                'description': f"Comprehensive introduction to {topic} concepts and applications"
            },
            {
                'title': f"Advanced {topic} Concepts",
                'provider': 'Khan Academy',
                'url': f"https://www.khanacademy.org/search?search_again=1&page_search_query={topic}",
                'difficulty': 'Intermediate',
                'description': f"Deep dive into advanced {topic} methodologies"
            }
        ]
        return courses

class ConceptExtractor:
    """Extracts key concepts from educational materials using Gemini"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def extract_key_concepts(self, content: str, subject: str) -> List[str]:
        """Extract key concepts using Gemini"""
        try:
            prompt = f"""
            Analyze the following educational content about {subject} and extract the 5-10 most important key concepts.
            Return only a JSON list of concepts, each as a short phrase or term.
            
            Content: {content[:2000]}  # Limit content length
            
            Format your response as a valid JSON array: ["concept1", "concept2", ...]
            """
            
            response = self.api_manager.gemini_model.generate_content(prompt)
            
            # Parse the JSON response
            concepts_text = response.text.strip()
            if concepts_text.startswith('```json'):
                concepts_text = concepts_text[7:-3]
            elif concepts_text.startswith('```'):
                concepts_text = concepts_text[3:-3]
                
            concepts = json.loads(concepts_text)
            return concepts if isinstance(concepts, list) else []
            
        except Exception as e:
            logger.error(f"Concept extraction error: {e}")
            # Fallback: simple keyword extraction
            words = content.lower().split()
            return list(set([word for word in words if len(word) > 5]))[:10]
    
    def determine_difficulty(self, content: str) -> str:
        """Determine content difficulty level"""
        try:
            prompt = f"""
            Analyze this educational content and determine its difficulty level.
            Respond with only one word: "Beginner", "Intermediate", or "Advanced"
            
            Content: {content[:1000]}
            """
            
            response = self.api_manager.gemini_model.generate_content(prompt)
            difficulty = response.text.strip()
            
            if difficulty in ["Beginner", "Intermediate", "Advanced"]:
                return difficulty
            return "Intermediate"  # Default
            
        except Exception as e:
            logger.error(f"Difficulty determination error: {e}")
            return "Intermediate"

class ContentGenerator:
    """Generates different types of educational content"""
    
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager
    
    def generate_quiz(self, material: LearningMaterial, num_questions: int = 5) -> Dict[str, Any]:
        """Generate quiz questions using Gemini"""
        try:
            prompt = f"""
            Create a {num_questions}-question multiple choice quiz based on this educational material about {material.subject}.
            
            Content: {material.content[:1500]}
            Key Concepts: {', '.join(material.key_concepts)}
            Difficulty: {material.difficulty_level}
            
            Format as JSON:
            {{
                "title": "Quiz Title",
                "questions": [
                    {{
                        "question": "Question text?",
                        "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
                        "correct_answer": "A",
                        "explanation": "Why this is correct"
                    }}
                ]
            }}
            """
            
            response = self.api_manager.gemini_model.generate_content(prompt)
            quiz_text = response.text.strip()
            
            # Clean JSON response
            if quiz_text.startswith('```json'):
                quiz_text = quiz_text[7:-3]
            elif quiz_text.startswith('```'):
                quiz_text = quiz_text[3:-3]
            
            return json.loads(quiz_text)
            
        except Exception as e:
            logger.error(f"Quiz generation error: {e}")
            return {"title": "Quiz", "questions": []}
    
    def generate_flashcards(self, material: LearningMaterial, num_cards: int = 10) -> Dict[str, Any]:
        """Generate flashcards using Gemini"""
        try:
            prompt = f"""
            Create {num_cards} flashcards based on this educational material about {material.subject}.
            
            Content: {material.content[:1500]}
            Key Concepts: {', '.join(material.key_concepts)}
            
            Format as JSON:
            {{
                "title": "Flashcard Set Title",
                "cards": [
                    {{
                        "front": "Question or term",
                        "back": "Answer or definition",
                        "concept": "related concept"
                    }}
                ]
            }}
            """
            
            response = self.api_manager.gemini_model.generate_content(prompt)
            cards_text = response.text.strip()
            
            # Clean JSON response
            if cards_text.startswith('```json'):
                cards_text = cards_text[7:-3]
            elif cards_text.startswith('```'):
                cards_text = cards_text[3:-3]
            
            return json.loads(cards_text)
            
        except Exception as e:
            logger.error(f"Flashcard generation error: {e}")
            return {"title": "Flashcards", "cards": []}
    
    def generate_summary(self, material: LearningMaterial) -> Dict[str, Any]:
        """Generate educational summary using Gemini"""
        try:
            prompt = f"""
            Create a comprehensive educational summary of this material about {material.subject}.
            Include key points, important concepts, and practical applications.
            
            Content: {material.content[:2000]}
            Key Concepts: {', '.join(material.key_concepts)}
            Difficulty: {material.difficulty_level}
            
            Format as JSON:
            {{
                "title": "Summary Title",
                "overview": "Brief overview paragraph",
                "key_points": ["point1", "point2", "point3"],
                "concepts_explained": {{
                    "concept1": "explanation1",
                    "concept2": "explanation2"
                }},
                "practical_applications": ["application1", "application2"],
                "further_reading": ["suggestion1", "suggestion2"]
            }}
            """
            
            response = self.api_manager.gemini_model.generate_content(prompt)
            summary_text = response.text.strip()
            
            # Clean JSON response
            if summary_text.startswith('```json'):
                summary_text = summary_text[7:-3]
            elif summary_text.startswith('```'):
                summary_text = summary_text[3:-3]
            
            return json.loads(summary_text)
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return {"title": "Summary", "overview": "", "key_points": []}

class ReasoningEngine:
    """Decision-making engine that connects user requests with appropriate actions"""
    
    def __init__(self, api_manager: APIManager, concept_extractor: ConceptExtractor, content_generator: ContentGenerator):
        self.api_manager = api_manager
        self.concept_extractor = concept_extractor
        self.content_generator = content_generator
    
    def analyze_user_request(self, user_input: str) -> Dict[str, Any]:
        """Analyze user request and determine appropriate actions"""
        try:
            prompt = f"""
            Analyze this user request for educational content and extract:
            1. Subject/topic
            2. Preferred content types (quiz, flashcards, summary, video recommendations)
            3. Difficulty level preference
            4. Specific requirements
            
            User request: {user_input}
            
            Respond in JSON format:
            {{
                "subject": "identified subject",
                "content_types": ["quiz", "flashcards", "summary"],
                "difficulty": "Beginner/Intermediate/Advanced",
                "specific_requirements": ["requirement1", "requirement2"]
            }}
            """
            
            response = self.api_manager.gemini_model.generate_content(prompt)
            analysis_text = response.text.strip()
            
            # Clean JSON response
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text[7:-3]
            elif analysis_text.startswith('```'):
                analysis_text = analysis_text[3:-3]
            
            return json.loads(analysis_text)
            
        except Exception as e:
            logger.error(f"Request analysis error: {e}")
            return {
                "subject": "general",
                "content_types": ["summary"],
                "difficulty": "Intermediate",
                "specific_requirements": []
            }
    
    def recommend_learning_path(self, subject: str, current_level: str) -> Dict[str, Any]:
        """Recommend a personalized learning path"""
        try:
            # Get various resources
            wikipedia_content = self.api_manager.get_wikipedia_content(subject)
            youtube_videos = self.api_manager.get_youtube_educational_videos(subject)
            books = self.api_manager.get_open_library_books(subject.lower().replace(' ', '_'))
            courses = self.api_manager.get_free_course_content(subject)
            
            return {
                "subject": subject,
                "current_level": current_level,
                "wikipedia_reference": wikipedia_content,
                "recommended_videos": youtube_videos[:3],
                "recommended_books": books[:3],
                "recommended_courses": courses,
                "next_steps": [
                    f"Start with basic concepts in {subject}",
                    f"Practice with interactive exercises",
                    f"Apply knowledge through projects"
                ]
            }
            
        except Exception as e:
            logger.error(f"Learning path recommendation error: {e}")
            return {"subject": subject, "recommendations": []}

class EducationalAIAgent:
    """Main agent class that orchestrates all components"""
    
    def __init__(self):
        self.api_manager = APIManager()
        self.concept_extractor = ConceptExtractor(self.api_manager)
        self.content_generator = ContentGenerator(self.api_manager)
        self.reasoning_engine = ReasoningEngine(
            self.api_manager, 
            self.concept_extractor, 
            self.content_generator
        )
        
        # Storage for generated content
        self.content_history: List[GeneratedContent] = []
    
    def process_user_request(self, user_input: str, uploaded_content: str = None) -> Dict[str, Any]:
        """Main method to process user requests and generate educational content"""
        
        # Analyze user request
        request_analysis = self.reasoning_engine.analyze_user_request(user_input)
        
        # Get or create learning material
        if uploaded_content:
            content = uploaded_content
            source = "User Upload"
        else:
            # Get content from Wikipedia
            wiki_content = self.api_manager.get_wikipedia_content(request_analysis['subject'])
            content = wiki_content.get('extract', '')
            source = wiki_content.get('source', 'Generated')
        
        if not content:
            return {"error": "No content available for this topic"}
        
        # Extract key concepts
        key_concepts = self.concept_extractor.extract_key_concepts(
            content, request_analysis['subject']
        )
        
        # Determine difficulty
        difficulty = self.concept_extractor.determine_difficulty(content)
        
        # Create learning material object
        material = LearningMaterial(
            title=f"Learning Material: {request_analysis['subject']}",
            content=content,
            source=source,
            difficulty_level=difficulty,
            subject=request_analysis['subject'],
            key_concepts=key_concepts
        )
        
        # Generate requested content types
        generated_content = {}
        
        for content_type in request_analysis['content_types']:
            if content_type == 'quiz':
                quiz = self.content_generator.generate_quiz(material)
                generated_content['quiz'] = quiz
                
                # Store in history
                self.content_history.append(GeneratedContent(
                    content_type=ContentType.QUIZ,
                    title=quiz.get('title', 'Quiz'),
                    content=quiz,
                    metadata={'subject': material.subject, 'difficulty': material.difficulty_level},
                    created_at=datetime.now()
                ))
                
            elif content_type == 'flashcards':
                flashcards = self.content_generator.generate_flashcards(material)
                generated_content['flashcards'] = flashcards
                
                # Store in history
                self.content_history.append(GeneratedContent(
                    content_type=ContentType.FLASHCARD,
                    title=flashcards.get('title', 'Flashcards'),
                    content=flashcards,
                    metadata={'subject': material.subject, 'difficulty': material.difficulty_level},
                    created_at=datetime.now()
                ))
                
            elif content_type == 'summary':
                summary = self.content_generator.generate_summary(material)
                generated_content['summary'] = summary
                
                # Store in history
                self.content_history.append(GeneratedContent(
                    content_type=ContentType.SUMMARY,
                    title=summary.get('title', 'Summary'),
                    content=summary,
                    metadata={'subject': material.subject, 'difficulty': material.difficulty_level},
                    created_at=datetime.now()
                ))
        
        # Get learning path recommendations
        learning_path = self.reasoning_engine.recommend_learning_path(
            request_analysis['subject'], 
            difficulty
        )
        
        return {
            'request_analysis': request_analysis,
            'material': {
                'title': material.title,
                'subject': material.subject,
                'difficulty': material.difficulty_level,
                'key_concepts': material.key_concepts,
                'source': material.source
            },
            'generated_content': generated_content,
            'learning_path': learning_path,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_content_analytics(self) -> Dict[str, Any]:
        """Get analytics about generated content"""
        if not self.content_history:
            return {"message": "No content generated yet"}
        
        # Analyze content types
        content_types = [content.content_type.value for content in self.content_history]
        subjects = [content.metadata.get('subject', 'Unknown') for content in self.content_history]
        
        analytics = {
            'total_content_generated': len(self.content_history),
            'content_type_distribution': {
                ct: content_types.count(ct) for ct in set(content_types)
            },
            'subject_distribution': {
                subject: subjects.count(subject) for subject in set(subjects)
            },
            'recent_activity': [
                {
                    'title': content.title,
                    'type': content.content_type.value,
                    'created': content.created_at.strftime('%Y-%m-%d %H:%M')
                }
                for content in sorted(self.content_history, key=lambda x: x.created_at, reverse=True)[:5]
            ]
        }
        
        return analytics

# Example usage and testing
if __name__ == "__main__":
    # Initialize the agent
    agent = EducationalAIAgent()
    
    # Example user requests
    test_requests = [
        "I want to learn about machine learning. Can you create a quiz and flashcards?",
        "Generate a summary about photosynthesis for high school level",
        "Create study materials for calculus - I need flashcards and practice questions"
    ]
    
    for request in test_requests:
        print(f"\n{'='*50}")
        print(f"Processing: {request}")
        print('='*50)
        
        try:
            result = agent.process_user_request(request)
            
            print(f"Subject: {result['material']['subject']}")
            print(f"Difficulty: {result['material']['difficulty']}")
            print(f"Key Concepts: {', '.join(result['material']['key_concepts'][:3])}")
            print(f"Generated Content Types: {list(result['generated_content'].keys())}")
            
            # Show analytics
            analytics = agent.get_content_analytics()
            print(f"Total Content Generated: {analytics['total_content_generated']}")
            
        except Exception as e:
            print(f"Error processing request: {e}")
    
    print(f"\n{'='*50}")
    print("Final Analytics:")
    print('='*50)
    final_analytics = agent.get_content_analytics()
    print(json.dumps(final_analytics, indent=2, default=str))
