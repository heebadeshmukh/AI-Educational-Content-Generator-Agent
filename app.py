# Streamlit Web Interface for Educational AI Agent
# Run with: streamlit run app.py

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Import the main agent (assuming the previous code is in educational_agent.py)
# from educational_agent import EducationalAIAgent, ContentType

# For demo purposes, we'll include a simplified version here
class StreamlitEducationalAgent:
    """Simplified version for Streamlit demo"""
    
    def __init__(self):
        self.content_history = []
    
    def process_request(self, user_input, content_types, difficulty):
        """Simulate processing a user request"""
        # Simulate processing time
        time.sleep(1)
        
        # Mock response
        response = {
            'material': {
                'subject': user_input.split()[-1] if user_input.split() else 'General',
                'difficulty': difficulty,
                'key_concepts': ['Concept 1', 'Concept 2', 'Concept 3'],
                'source': 'Wikipedia'
            },
            'generated_content': {}
        }
        
        # Generate mock content based on selected types
        if 'Quiz' in content_types:
            response['generated_content']['quiz'] = {
                'title': f'Quiz: {response["material"]["subject"]}',
                'questions': [
                    {
                        'question': f'What is the main concept of {response["material"]["subject"]}?',
                        'options': ['A) Option 1', 'B) Option 2', 'C) Option 3', 'D) Option 4'],
                        'correct_answer': 'A',
                        'explanation': 'This is the correct answer because...'
                    }
                ]
            }
        
        if 'Flashcards' in content_types:
            response['generated_content']['flashcards'] = {
                'title': f'Flashcards: {response["material"]["subject"]}',
                'cards': [
                    {
                        'front': f'What is {response["material"]["subject"]}?',
                        'back': f'{response["material"]["subject"]} is an important concept that...',
                        'concept': 'Basic Definition'
                    }
                ]
            }
        
        if 'Summary' in content_types:
            response['generated_content']['summary'] = {
                'title': f'Summary: {response["material"]["subject"]}',
                'overview': f'This summary covers the key aspects of {response["material"]["subject"]}.',
                'key_points': ['Point 1', 'Point 2', 'Point 3'],
                'concepts_explained': {
                    'Concept 1': 'Explanation of concept 1',
                    'Concept 2': 'Explanation of concept 2'
                }
            }
        
        # Add to history
        self.content_history.append({
            'timestamp': datetime.now(),
            'subject': response['material']['subject'],
            'content_types': content_types,
            'difficulty': difficulty
        })
        
        return response

# Initialize the agent
if 'agent' not in st.session_state:
    st.session_state.agent = StreamlitEducationalAgent()

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
</style>
""", unsafe_allow_html=True)

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
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", 
                                  help="Enter your Google Gemini API key")
youtube_key = st.sidebar.text_input("YouTube API Key", type="password",
                                   help="Enter your YouTube Data API key")

if gemini_key:
    st.sidebar.success("✅ Gemini API configured")
if youtube_key:
    st.sidebar.success("✅ YouTube API configured")

st.sidebar.markdown("---")

# Quick Stats
if st.session_state.agent.content_history:
    st.sidebar.subheader("📊 Quick Stats")
    total_content = len(st.session_state.agent.content_history)
    subjects = [item['subject'] for item in st.session_state.agent.content_history]
    unique_subjects = len(set(subjects))
    
    st.sidebar.metric("Total Content Generated", total_content)
    st.sidebar.metric("Unique Subjects", unique_subjects)
    
    # Recent activity
    st.sidebar.subheader("🕒 Recent Activity")
    for item in st.session_state.agent.content_history[-3:]:
        st.sidebar.text(f"• {item['subject']} ({item['difficulty']})")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Generate Content", "📚 Content Library", "📈 Analytics", "🛠️ Settings"])

with tab1:
    st.header("Generate Educational Content")
    
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
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    with col2:
        # Content type selection
        st.subheader("Content Types")
        content_types = st.multiselect(
            "Select content types to generate:",
            ["Quiz", "Flashcards", "Summary", "Video Recommendations"],
            default=["Summary"]
        )
        
        # Difficulty selection
        difficulty = st.selectbox(
            "Difficulty Level:",
            ["Beginner", "Intermediate", "Advanced"],
            index=1
        )
        
        # Number of items
        if "Quiz" in content_types:
            num_questions = st.slider("Number of Quiz Questions", 3, 15, 5)
        if "Flashcards" in content_types:
            num_flashcards = st.slider("Number of Flashcards", 5, 25, 10)
    
    # Generate button
    if st.button("🚀 Generate Learning Materials", type="primary", use_container_width=True):
        if user_input.strip() or uploaded_file:
            with st.spinner("🤖 AI is creating your personalized learning materials..."):
                try:
                    # Process the request
                    response = st.session_state.agent.process_request(
                        user_input, content_types, difficulty
                    )
                    
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
                            st.markdown(f"### {quiz['title']}")
                            
                            for i, q in enumerate(quiz['questions'], 1):
                                st.markdown(f"""
                                <div class="quiz-question">
                                    <strong>Question {i}:</strong> {q['question']}<br><br>
                                    {'<br>'.join(q['options'])}<br><br>
                                    <strong>Correct Answer:</strong> {q['correct_answer']}<br>
                                    <strong>Explanation:</strong> {q['explanation']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Flashcards
                    if 'flashcards' in response['generated_content']:
                        with st.expander("🎴 Flashcards", expanded=True):
                            flashcards = response['generated_content']['flashcards']
                            st.markdown(f"### {flashcards['title']}")
                            
                            for i, card in enumerate(flashcards['cards'], 1):
                                st.markdown(f"""
                                <div class="flashcard">
                                    <strong>Card {i} - Front:</strong> {card['front']}<br>
                                    <strong>Back:</strong> {card['back']}<br>
                                    <em>Concept: {card['concept']}</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Summary
                    if 'summary' in response['generated_content']:
                        with st.expander("📄 Summary", expanded=True):
                            summary = response['generated_content']['summary']
                            st.markdown(f"### {summary['title']}")
                            
                            st.markdown(f"""
                            <div class="summary-section">
                                <h4>Overview</h4>
                                <p>{summary['overview']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if 'key_points' in summary:
                                st.markdown("#### Key Points")
                                for point in summary['key_points']:
                                    st.write(f"• {point}")
                            
                            if 'concepts_explained' in summary:
                                st.markdown("#### Concepts Explained")
                                for concept, explanation in summary['concepts_explained'].items():
                                    st.write(f"**{concept}:** {explanation}")
                
                except Exception as e:
                    st.error(f"❌ Error generating content: {str(e)}")
        else:
            st.warning("⚠️ Please enter a topic or upload a file to get started!")

with tab2:
    st.header("📚 Content Library")
    
    if st.session_state.agent.content_history:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            subjects = list(set([item['subject'] for item in st.session_state.agent.content_history]))
            selected_subject = st.selectbox("Filter by Subject", ["All"] + subjects)
        
        with col2:
            difficulties = list(set([item['difficulty'] for item in st.session_state.agent.content_history]))
            selected_difficulty = st.selectbox("Filter by Difficulty", ["All"] + difficulties)
        
        with col3:
            content_type_filter = st.multiselect(
                "Filter by Content Type", 
                ["Quiz", "Flashcards", "Summary", "Video Recommendations"]
            )
        
        # Display filtered content
        filtered_history = st.session_state.agent.content_history.copy()
        
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
                
                st.divider()
    else:
        st.info("📭 No content generated yet. Go to the 'Generate Content' tab to create your first learning materials!")

with tab3:
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.agent.content_history:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_content = len(st.session_state.agent.content_history)
        subjects = [item['subject'] for item in st.session_state.agent.content_history]
        unique_subjects = len(set(subjects))
        
        # Calculate average difficulty
        difficulty_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
        difficulties = [difficulty_map[item['difficulty']] for item in st.session_state.agent.content_history]
        avg_difficulty = sum(difficulties) / len(difficulties)
        avg_difficulty_text = ["Beginner", "Intermediate", "Advanced"][round(avg_difficulty) - 1]
        
        # Most popular content type
        all_content_types = []
        for item in st.session_state.agent.content_history:
            all_content_types.extend(item['content_types'])
        
        from collections import Counter
        content_type_counts = Counter(all_content_types)
        most_popular_type = content_type_counts.most_common(1)[0][0] if content_type_counts else "None"
        
        with col1:
            st.metric("Total Content", total_content)
        with col2:
            st.metric("Unique Subjects", unique_subjects)
        with col3:
            st.metric("Avg. Difficulty", avg_difficulty_text)
        with col4:
            st.metric("Popular Type", most_popular_type)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Subject distribution
            subject_counts = Counter(subjects)
            if subject_counts:
                fig_subjects = px.pie(
                    values=list(subject_counts.values()),
                    names=list(subject_counts.keys()),
                    title="Content by Subject"
                )
                st.plotly_chart(fig_subjects, use_container_width=True)
        
        with col2:
            # Content type distribution
            if content_type_counts:
                fig_types = px.bar(
                    x=list(content_type_counts.keys()),
                    y=list(content_type_counts.values()),
                    title="Content Type Distribution"
                )
                st.plotly_chart(fig_types, use_container_width=True)
        
        # Timeline
        st.subheader("📅 Content Generation Timeline")
        
        # Group by date
        dates = [item['timestamp'].date() for item in st.session_state.agent.content_history]
        date_counts = Counter(dates)
        
        if date_counts:
            timeline_df = pd.DataFrame([
                {"Date": date, "Count": count} 
                for date, count in sorted(date_counts.items())
            ])
            
            fig_timeline = px.line(
                timeline_df, 
                x="Date", 
                y="Count", 
                title="Daily Content Generation",
                markers=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent activity table
        st.subheader("🕒 Recent Activity")
        recent_df = pd.DataFrame([
            {
                "Subject": item['subject'],
                "Content Types": ", ".join(item['content_types']),
                "Difficulty": item['difficulty'],
                "Timestamp": item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            }
            for item in reversed(st.session_state.agent.content_history[-10:])
        ])
        
        st.dataframe(recent_df, use_container_width=True)
        
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
           - Free tier: 60 requests per minute
        
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
    
    # Content Generation Settings
    st.subheader("⚙️ Content Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_difficulty = st.selectbox(
            "Default Difficulty Level",
            ["Beginner", "Intermediate", "Advanced"],
            index=1
        )
        
        default_quiz_questions = st.slider(
            "Default Quiz Questions",
            min_value=3,
            max_value=20,
            value=5
        )
    
    with col2:
        default_flashcards = st.slider(
            "Default Flashcards",
            min_value=5,
            max_value=30,
            value=10
        )
        
        auto_generate_videos = st.checkbox(
            "Auto-generate video recommendations",
            value=True
        )
    
    # Data Management
    st.subheader("💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Export History", use_container_width=True):
            if st.session_state.agent.content_history:
                # Create export data
                export_data = []
                for item in st.session_state.agent.content_history:
                    export_data.append({
                        'Subject': item['subject'],
                        'Content_Types': ', '.join(item['content_types']),
                        'Difficulty': item['difficulty'],
                        'Timestamp': item['timestamp'].isoformat()
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
            if st.session_state.agent.content_history:
                if st.button("⚠️ Confirm Clear", type="secondary"):
                    st.session_state.agent.content_history = []
                    st.success("History cleared!")
                    st.experimental_rerun()
            else:
                st.info("History is already empty")
    
    with col3:
        st.metric("Current Storage", f"{len(st.session_state.agent.content_history)} items")
    
    # About section
    st.subheader("ℹ️ About")
    st.markdown("""
    ### Educational AI Agent v1.0
    
    This application uses advanced AI to transform any educational topic into personalized learning materials:
    
    - **Quiz Generation**: Creates multiple-choice questions with explanations
    - **Flashcard Creation**: Generates front/back flashcard pairs
    - **Smart Summaries**: Produces structured overviews with key concepts
    - **Video Recommendations**: Finds relevant educational videos
    - **Learning Paths**: Suggests progression routes
    
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
    <a href="https://github.com" target="_blank">View Source Code</a>
</div>
""", unsafe_allow_html=True)
