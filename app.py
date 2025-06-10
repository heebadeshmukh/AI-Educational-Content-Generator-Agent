# Streamlit Web Interface for Educational AI Agent
# Run with: streamlit run app.py

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import logging

# Import the main agent
from main import EducationalAIAgent, ContentType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the agent
@st.cache_resource
def get_agent():
    """Initialize and cache the AI agent"""
    return EducationalAIAgent()

# Page configuration
st.set_page_config(
    page_title="Educational AI Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .content-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .quiz-question {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    .flashcard {
        background: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .summary-section {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #721c24;
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = get_agent()

if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎓 Educational AI Agent</h1>
    <p>Transform any topic into personalized learning materials with AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("---")

# API Configuration
st.sidebar.subheader("API Settings")

# Get API keys from environment or user input
gemini_key = st.sidebar.text_input(
    "Gemini API Key", 
    type="password", 
    value=os.getenv('GEMINI_API_KEY', ''),
    help="Enter your Google Gemini API key"
)

youtube_key = st.sidebar.text_input(
    "YouTube API Key", 
    type="password",
    value=os.getenv('YOUTUBE_API_KEY', ''),
    help="Enter your YouTube Data API key (optional)"
)

# Set environment variables if provided
if gemini_key:
    os.environ['GEMINI_API_KEY'] = gemini_key
    st.sidebar.success("✅ Gemini API configured")
else:
    st.sidebar.warning("⚠️ Gemini API key required")

if youtube_key:
    os.environ['YOUTUBE_API_KEY'] = youtube_key
    st.sidebar.success("✅ YouTube API configured")

st.sidebar.markdown("---")

# Quick Stats
if st.session_state.generation_history:
    st.sidebar.subheader("📊 Quick Stats")
    total_content = len(st.session_state.generation_history)
    subjects = [item['subject'] for item in st.session_state.generation_history]
    unique_subjects = len(set(subjects))
    
    st.sidebar.metric("Total Content Generated", total_content)
    st.sidebar.metric("Unique Subjects", unique_subjects)
    
    # Recent activity
    st.sidebar.subheader("🕒 Recent Activity")
    for item in st.session_state.generation_history[-3:]:
        st.sidebar.text(f"• {item['subject']} ({item['difficulty']})")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Generate Content", "📚 Content Library", "📈 Analytics", "🛠️ Settings"])

with tab1:
    st.header("Generate Educational Content")
    
    # Check if API key is configured
    if not gemini_key:
        st.error("🚨 Please configure your Gemini API key in the sidebar to use the AI features.")
        st.info("Get your free API key at: https://makersuite.google.com/app/apikey")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User input
        user_input = st.text_area(
            "What would you like to learn about?",
            placeholder="e.g., 'I want to learn about machine learning' or 'Create study materials for photosynthesis'",
            height=100
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Or upload your own study material",
            type=['txt', 'pdf', 'docx'],
            help="Upload documents to create personalized study materials"
        )
        
        uploaded_content = None
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Read file content
            try:
                if uploaded_file.type == "text/plain":
                    uploaded_content = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    # For PDF files, you'd need to implement PDF reading
                    st.warning("PDF processing not implemented yet. Please use text files.")
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    # For DOCX files, you'd need to implement DOCX reading
                    st.warning("DOCX processing not implemented yet. Please use text files.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        # Content type selection
        st.subheader("Content Types")
        content_types = st.multiselect(
            "Select content types to generate:",
            ["quiz", "flashcards", "summary"],
            default=["summary"],
            help="Choose which types of learning materials to create"
        )
        
        # Difficulty selection
        difficulty_override = st.selectbox(
            "Override Difficulty Level (optional):",
            ["Auto-detect", "Beginner", "Intermediate", "Advanced"],
            index=0,
            help="Leave as 'Auto-detect' to let AI determine the appropriate level"
        )
        
        # Additional options
        if "quiz" in content_types:
            num_questions = st.slider("Number of Quiz Questions", 3, 15, 5)
        if "flashcards" in content_types:
            num_flashcards = st.slider("Number of Flashcards", 5, 25, 10)
    
    # Generate button
    if st.button("🚀 Generate Learning Materials", type="primary", use_container_width=True):
        if user_input.strip() or uploaded_file:
            try:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🤖 Initializing AI agent...")
                progress_bar.progress(10)
                
                # Prepare the request
                request_text = user_input.strip() if user_input.strip() else f"Create learning materials for the uploaded content about {uploaded_file.name}"
                
                # Add content type preferences to the request
                if content_types:
                    content_type_text = ", ".join(content_types)
                    request_text += f". I want {content_type_text}."
                
                status_text.text("🔍 Analyzing your request...")
                progress_bar.progress(30)
                
                # Process the request using the real AI agent
                with st.spinner("🤖 AI is creating your personalized learning materials..."):
                    response = st.session_state.agent.process_user_request(
                        request_text, 
                        uploaded_content
                    )
                
                progress_bar.progress(100)
                status_text.text("✅ Content generated successfully!")
                
                # Check for errors
                if 'error' in response:
                    st.error(f"❌ Error: {response['error']}")
                else:
                    # Add to history
                    history_item = {
                        'timestamp': datetime.now(),
                        'subject': response['material']['subject'],
                        'content_types': content_types,
                        'difficulty': response['material']['difficulty'],
                        'response': response
                    }
                    st.session_state.generation_history.append(history_item)
                    
                    st.success("✅ Content generated successfully!")
                    
                    # Display material info
                    st.subheader("📋 Material Overview")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Subject", response['material']['subject'])
                    with col2:
                        st.metric("Difficulty", response['material']['difficulty'])
                    with col3:
                        st.metric("Key Concepts", len(response['material']['key_concepts']))
                    
                    st.info(f"**Source:** {response['material']['source']}")
                    
                    # Display key concepts
                    with st.expander("🔑 Key Concepts Identified"):
                        for i, concept in enumerate(response['material']['key_concepts'], 1):
                            st.write(f"{i}. {concept}")
                    
                    # Display generated content
                    st.subheader("📚 Generated Content")
                    
                    # Quiz
                    if 'quiz' in response['generated_content']:
                        with st.expander("📝 Quiz", expanded=True):
                            quiz = response['generated_content']['quiz']
                            st.markdown(f"### {quiz.get('title', 'Quiz')}")
                            
                            if 'questions' in quiz and quiz['questions']:
                                for i, q in enumerate(quiz['questions'], 1):
                                    st.markdown(f"""
                                    <div class="quiz-question">
                                        <strong>Question {i}:</strong> {q.get('question', 'No question text')}<br><br>
                                        {'<br>'.join(q.get('options', []))}<br><br>
                                        <strong>Correct Answer:</strong> {q.get('correct_answer', 'N/A')}<br>
                                        <strong>Explanation:</strong> {q.get('explanation', 'No explanation provided')}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("No quiz questions were generated.")
                    
                    # Flashcards
                    if 'flashcards' in response['generated_content']:
                        with st.expander("🎴 Flashcards", expanded=True):
                            flashcards = response['generated_content']['flashcards']
                            st.markdown(f"### {flashcards.get('title', 'Flashcards')}")
                            
                            if 'cards' in flashcards and flashcards['cards']:
                                for i, card in enumerate(flashcards['cards'], 1):
                                    st.markdown(f"""
                                    <div class="flashcard">
                                        <strong>Card {i} - Front:</strong> {card.get('front', 'No front text')}<br>
                                        <strong>Back:</strong> {card.get('back', 'No back text')}<br>
                                        <em>Concept: {card.get('concept', 'General')}</em>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("No flashcards were generated.")
                    
                    # Summary
                    if 'summary' in response['generated_content']:
                        with st.expander("📄 Summary", expanded=True):
                            summary = response['generated_content']['summary']
                            st.markdown(f"### {summary.get('title', 'Summary')}")
                            
                            if summary.get('overview'):
                                st.markdown(f"""
                                <div class="summary-section">
                                    <h4>Overview</h4>
                                    <p>{summary['overview']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            if 'key_points' in summary and summary['key_points']:
                                st.markdown("#### Key Points")
                                for point in summary['key_points']:
                                    st.write(f"• {point}")
                            
                            if 'concepts_explained' in summary and summary['concepts_explained']:
                                st.markdown("#### Concepts Explained")
                                for concept, explanation in summary['concepts_explained'].items():
                                    st.write(f"**{concept}:** {explanation}")
                            
                            if 'practical_applications' in summary and summary['practical_applications']:
                                st.markdown("#### Practical Applications")
                                for app in summary['practical_applications']:
                                    st.write(f"• {app}")
                    
                    # Learning Path Recommendations
                    if 'learning_path' in response:
                        with st.expander("🎯 Learning Path Recommendations"):
                            learning_path = response['learning_path']
                            
                            # Wikipedia reference
                            if 'wikipedia_reference' in learning_path and learning_path['wikipedia_reference']:
                                st.markdown("#### 📖 Reference Material")
                                wiki = learning_path['wikipedia_reference']
                                st.write(f"**{wiki.get('title', 'Wikipedia Article')}**")
                                if wiki.get('url'):
                                    st.write(f"[Read more on Wikipedia]({wiki['url']})")
                            
                            # Video recommendations
                            if 'recommended_videos' in learning_path and learning_path['recommended_videos']:
                                st.markdown("#### 🎥 Recommended Videos")
                                for video in learning_path['recommended_videos']:
                                    st.write(f"• [{video.get('title', 'Video')}]({video.get('url', '#')}) - {video.get('channel', 'Unknown Channel')}")
                            
                            # Book recommendations
                            if 'recommended_books' in learning_path and learning_path['recommended_books']:
                                st.markdown("#### 📚 Recommended Books")
                                for book in learning_path['recommended_books']:
                                    authors = ', '.join(book.get('authors', ['Unknown Author']))
                                    st.write(f"• **{book.get('title', 'Book')}** by {authors}")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error generating content: {str(e)}")
                logger.error(f"Content generation error: {e}")
                
                # Show detailed error information
                with st.expander("🔍 Error Details"):
                    st.code(str(e))
                    st.write("**Troubleshooting Tips:**")
                    st.write("1. Check your Gemini API key is valid")
                    st.write("2. Ensure you have internet connection")
                    st.write("3. Try with a simpler topic")
                    st.write("4. Check if you've exceeded API limits")
        else:
            st.warning("⚠️ Please enter a topic or upload a file to get started!")

with tab2:
    st.header("📚 Content Library")
    
    if st.session_state.generation_history:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            subjects = list(set([item['subject'] for item in st.session_state.generation_history]))
            selected_subject = st.selectbox("Filter by Subject", ["All"] + subjects)
        
        with col2:
            difficulties = list(set([item['difficulty'] for item in st.session_state.generation_history]))
            selected_difficulty = st.selectbox("Filter by Difficulty", ["All"] + difficulties)
        
        with col3:
            all_content_types = list(set([ct for item in st.session_state.generation_history for ct in item['content_types']]))
            content_type_filter = st.multiselect(
                "Filter by Content Type", 
                all_content_types
            )
        
        # Display filtered content
        filtered_history = st.session_state.generation_history.copy()
        
        if selected_subject != "All":
            filtered_history = [item for item in filtered_history if item['subject'] == selected_subject]
        
        if selected_difficulty != "All":
            filtered_history = [item for item in filtered_history if item['difficulty'] == selected_difficulty]
        
        if content_type_filter:
            filtered_history = [item for item in filtered_history 
                              if any(ct in item['content_types'] for ct in content_type_filter)]
        
        # Display results
        st.write(f"**Found {len(filtered_history)} items**")
        
        for i, item in enumerate(reversed(filtered_history)):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.write(f"**{item['subject']}**")
                
                with col2:
                    st.write(f"Difficulty: {item['difficulty']}")
                
                with col3:
                    st.write(f"Types: {', '.join(item['content_types'])}")
                
                with col4:
                    st.write(item['timestamp'].strftime("%m/%d %H:%M"))
                
                # Add expand button to view content
                if st.button(f"View Details", key=f"view_{i}"):
                    st.json(item['response'])
                
                st.divider()
    else:
        st.info("📭 No content generated yet. Go to the 'Generate Content' tab to create your first learning materials!")

with tab3:
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.generation_history:
        # Get analytics from the agent
        try:
            analytics = st.session_state.agent.get_content_analytics()
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Content", analytics.get('total_content_generated', 0))
            with col2:
                st.metric("Content Types", len(analytics.get('content_type_distribution', {})))
            with col3:
                st.metric("Subjects Covered", len(analytics.get('subject_distribution', {})))
            with col4:
                recent_count = len(analytics.get('recent_activity', []))
                st.metric("Recent Items", recent_count)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Subject distribution
                subject_dist = analytics.get('subject_distribution', {})
                if subject_dist:
                    fig_subjects = px.pie(
                        values=list(subject_dist.values()),
                        names=list(subject_dist.keys()),
                        title="Content by Subject"
                    )
                    st.plotly_chart(fig_subjects, use_container_width=True)
            
            with col2:
                # Content type distribution
                content_type_dist = analytics.get('content_type_distribution', {})
                if content_type_dist:
                    fig_types = px.bar(
                        x=list(content_type_dist.keys()),
                        y=list(content_type_dist.values()),
                        title="Content Type Distribution"
                    )
                    st.plotly_chart(fig_types, use_container_width=True)
            
            # Recent activity
            st.subheader("🕒 Recent Activity")
            recent_activity = analytics.get('recent_activity', [])
            if recent_activity:
                activity_df = pd.DataFrame(recent_activity)
                st.dataframe(activity_df, use_container_width=True)
            else:
                st.info("No recent activity to display")
                
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
            # Fallback to session state analytics
            total_content = len(st.session_state.generation_history)
            subjects = [item['subject'] for item in st.session_state.generation_history]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Content", total_content)
            with col2:
                st.metric("Unique Subjects", len(set(subjects)))
        
    else:
        st.info("📊 No data available yet. Generate some content to see analytics!")

with tab4:
    st.header("🛠️ Settings & Configuration")
    
    # API Settings
    st.subheader("🔌 API Configuration")
    
    with st.expander("API Keys & Setup", expanded=True):
        st.markdown("""
        ### Required APIs (Free Tiers Available)
        
        1. **Google Gemini API** (Required)
           - Get your free API key at: https://makersuite.google.com/app/apikey
           - Free tier: 15 requests per minute
        
        2. **YouTube Data API v3** (Optional)
           - Get your API key at: https://console.developers.google.com/
           - Free tier: 10,000 requests per day
        
        3. **Wikipedia API** (Free)
           - No API key required
           - Rate limit: 200 requests per second
        
        4. **Open Library API** (Free)
           - No API key required
           - Rate limit: 100 requests per second
        """)
        
        # Test API connections
        if st.button("🧪 Test API Connections"):
            test_results = {}
            
            # Test Gemini API
            if gemini_key:
                try:
                    # Simple test request
                    test_agent = EducationalAIAgent()
                    result = test_agent.reasoning_engine.analyze_user_request("test")
                    test_results['Gemini API'] = "✅ Working"
                except Exception as e:
                    test_results['Gemini API'] = f"❌ Error: {str(e)}"
            else:
                test_results['Gemini API'] = "⚠️ No API key provided"
            
            # Test Wikipedia API
            try:
                import requests
                response = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/test", timeout=5)
                if response.status_code == 200:
                    test_results['Wikipedia API'] = "✅ Working"
                else:
                    test_results['Wikipedia API'] = f"❌ Status: {response.status_code}"
            except Exception as e:
                test_results['Wikipedia API'] = f"❌ Error: {str(e)}"
            
            # Display results
            for api, status in test_results.items():
                st.write(f"**{api}:** {status}")
    
    # Content Generation Settings
    st.subheader("⚙️ Content Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_content_types = st.multiselect(
            "Default Content Types",
            ["quiz", "flashcards", "summary"],
            default=["summary"]
        )
        
        max_concepts = st.slider(
            "Maximum Key Concepts to Extract",
            min_value=5,
            max_value=20,
            value=10
        )
    
    with col2:
        generation_timeout = st.slider(
            "Generation Timeout (seconds)",
            min_value=30,
            max_value=300,
            value=120
        )
        
        enable_debug = st.checkbox(
            "Enable Debug Mode",
            value=False,
            help="Show detailed logging information"
        )
    
    # Data Management
    st.subheader("💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Export History", use_container_width=True):
            if st.session_state.generation_history:
                # Create export data
                export_data = []
                for item in st.session_state.generation_history:
                    export_data.append({
                        'Subject': item['subject'],
                        'Content_Types': ', '.join(item['content_types']),
                        'Difficulty': item['difficulty'],
                        'Timestamp': item['timestamp'].isoformat(),
                        'Key_Concepts': ', '.join(item['response']['material']['key_concepts'][:5])
                    })
                
                # Convert to CSV
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📁 Download CSV",
                    data=csv,
                    file_name=f"educational_content_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No history to export")
    
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            if st.session_state.generation_history:
                if st.button("⚠️ Confirm Clear", type="secondary"):
                    st.session_state.generation_history = []
                    st.success("History cleared!")
                    st.rerun()
            else:
                st.info("History is already empty")
    
    with col3:
        st.metric("Current Storage", f"{len(st.session_state.generation_history)} items")
    
    # Agent Status
    st.subheader("🤖 Agent Status")
    try:
        agent_analytics = st.session_state.agent.get_content_analytics()
        st.success("✅ Agent is operational")
        st.write(f"Agent has generated {agent_analytics.get('total_content_generated', 0)} pieces of content")
    except Exception as e:
        st.error(f"❌ Agent error: {e}")
    
    # About section
    st.subheader("ℹ️ About")
    st.markdown("""
    ### Educational AI Agent v1.0
    
    This application uses advanced AI to transform any educational topic into personalized learning materials:
    
    - **Quiz Generation**: Creates multiple-choice questions with explanations
    - **Flashcard Creation**: Generates front/back flashcard pairs
    - **Smart Summaries**: Produces structured overviews with key concepts
    - **Learning Paths**: Suggests progression routes with multimedia resources
    
    **Technologies Used:**
    - Google Gemini Pro for content generation
    - LangChain for processing pipelines
    - Multiple free educational APIs
    - Streamlit for the web interface
    
    **Privacy & Data:**
    - All processing happens in real-time
    - No personal data is stored permanently
    - Content history is session-based only
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🎓 Educational AI Agent | Built with Streamlit & Google Gemini | 
    <a href="https://github.com" target="_blank">Powered by AI</a>
</div>
""", unsafe_allow_html=True)
