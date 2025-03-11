import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from openai import OpenAI
import json
import re

# Set page configuration
st.set_page_config(
    page_title="Talk to Your Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force dark theme
st.markdown("""
    <script>
        var elements = window.parent.document.querySelectorAll('.stApp')
        elements[0].style.background = '#121212';
    </script>
    """, unsafe_allow_html=True)

# Sample datasets
@st.cache_data
def load_iris():
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = [data.target_names[i] for i in data.target]
    return df

@st.cache_data
def load_boston_housing():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['PRICE'] = data.target
    return df

@st.cache_data
def load_titanic():
    # Using a simplified version of the Titanic dataset
    data = {
        'PassengerId': range(1, 101),
        'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1] * 10,
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2] * 10,
        'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina', 
                'Futrelle, Mrs. Jacques Heath', 'Allen, Mr. William Henry', 'Moran, Mr. James', 
                'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard', 'Johnson, Mrs. Oscar W', 
                'Nasser, Mrs. Nicholas'] * 10,
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'] * 10,
        'Age': [22, 38, 26, 35, 35, None, 54, 2, 27, 14] * 10,
        'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1] * 10,
        'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0] * 10,
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07] * 10
    }
    return pd.DataFrame(data)

# LLM helper functions
def query_llm(prompt, api_key, model="gpt-3.5-turbo"):
    if not api_key:
        return "Please enter an OpenAI API key to use the LLM features."
    
    client = OpenAI(api_key=api_key)
    try:
        system_message = """You are an expert data analysis assistant that helps users understand their data through clear explanations and visualizations.

GUIDELINES FOR YOUR RESPONSES:
1. Provide thoughtful and accurate analysis based on the data provided
2. Use clear, concise language that non-technical users can understand
3. When creating visualizations:
   - Choose the most appropriate chart type for the data and question
   - Use pleasing color schemes (prefer using seaborn's 'colorblind' palette when applicable)
   - Ensure all elements are properly labeled (axis, titles, legends)
   - Add grid lines for better readability
   - Set appropriate figure size for clarity
4. Format your explanations with headers and bullet points when helpful
5. Mention limitations or caveats in your analysis when relevant
6. Always provide actionable insights based on the data

Your responses should contain both text explanation and visualization code when appropriate."""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying OpenAI API: {str(e)}"

def extract_code(llm_response):
    code_pattern = r'```python(.*?)```'
    code_matches = re.findall(code_pattern, llm_response, re.DOTALL)
    if code_matches:
        return code_matches[0].strip()
    return None

def execute_plot_code(code, df):
    # Create a safe namespace with only necessary modules and data
    namespace = {
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'df': df,
        'io': io,
        'base64': base64,
        'np': __import__('numpy')
    }
    
    try:
        # Add style configuration for dark mode compatible visualizations
        style_code = """
# Use a light background for plots with dark text for better readability
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)  # Increased figure size for better visibility
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

# Set seaborn style with clean white background and improved contrast
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.4)  # Increased font scale

# Vibrant color palette with good contrast
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
sns.set_palette(sns.color_palette(colors))
"""
        
        # Combine style code with user code
        full_code = style_code + "\n" + code
        
        # Add DPI setting for higher resolution output
        if 'plt.show()' in full_code:
            full_code = full_code.replace('plt.show()', 'plt.tight_layout()')
        if 'plt.' in full_code and not full_code.strip().endswith('plt.tight_layout()'):
            full_code += '\nplt.tight_layout()'
        
        # Execute the code
        exec(full_code, namespace)
        
        # Save the figure to a bytes buffer with higher DPI and improved visibility
        buf = io.BytesIO()
        # Make sure figure has clear background and contrasting foreground elements
        for ax in plt.gcf().get_axes():
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.title.set_color('black')
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color('black')
            if hasattr(ax, 'get_legend') and ax.get_legend() is not None:
                for text in ax.get_legend().get_texts():
                    text.set_color('black')
        
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')  # Higher DPI and white background
        buf.seek(0)
        plt.close()
        
        # Encode the bytes to base64 for displaying
        data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return data
    except Exception as e:
        return str(e)

# Custom CSS with dark mode compatibility but using native Streamlit elements where possible
st.markdown("""
<style>
    /* Set page background and text colors for dark mode */
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2196F3;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2196F3;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #333;
        padding-bottom: 0.5rem;
    }
    
    /* Buttons and basic UI components */
    .stButton>button {
        background-color: #2196F3;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #1976D2;
    }
    
    .stTextArea>div>div>textarea {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 10px;
        background-color: #2a2a2a;
        color: #f0f0f0;
    }
    
    /* Make sure text in dataframes is visible */
    .dataframe {
        color: #f0f0f0 !important;
    }
    
    .dataframe th {
        background-color: #2a2a2a !important;
        color: #f0f0f0 !important;
    }
    
    .dataframe td {
        background-color: #1e1e1e !important;
        color: #f0f0f0 !important;
    }
    
    /* Fix expander text colors */
    .streamlit-expanderHeader {
        color: #f0f0f0 !important;
    }
    
    /* Fix select box text colors */
    .stSelectbox > div > div {
        color: #f0f0f0 !important;
    }
    
    /* Fix widget label colors */
    .stTextInput > label, .stSelectbox > label, .stTextArea > label {
        color: #e0e0e0 !important;
    }
    
    /* Custom visualization container */
    .visualization-container {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Make example buttons stand out more */
    .example-button {
        margin-bottom: 10px !important;
        background-color: #263238 !important;
        color: #f0f0f0 !important;
        font-weight: 500 !important;
    }
    
    /* Make sure images have proper contrast with background */
    .stImage img {
        background-color: #ffffff !important;
        border: 1px solid #aaa !important;
        border-radius: 10px !important;
        padding: 15px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state to store the selected question
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""

# Main app
st.markdown('<div class="main-header">Talk to Your Data 📊</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your data and get instant insights with AI assistance.</div>', unsafe_allow_html=True)

# Sidebar for dataset selection and API key
with st.sidebar:
    st.markdown("<div style='text-align: center; font-weight: 600; font-size: 1.5rem; margin-bottom: 20px; color: #2196F3;'>Data Assistant</div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #e0e0e0;'>🛠️ Settings</h3>", unsafe_allow_html=True)
    
    dataset_option = st.selectbox(
        "📊 Choose a dataset:",
        ("Iris Flowers", "California Housing", "Titanic")
    )
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # OpenAI model selection
    openai_model = st.selectbox(
        "🤖 Select OpenAI model:",
        ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o")
    )
    
    api_key = st.text_input("🔑 Enter your OpenAI API key:", type="password")
    
    st.markdown("<hr style='margin: 20px 0; border-color: #333;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #e0e0e0;'>ℹ️ About</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background-color: #2a2a2a; padding: 15px; border-radius: 10px; color: #f0f0f0;'>
    This app lets you interact with data using natural language. Simply:
    <ol>
        <li>Select a dataset</li>
        <li>Choose an OpenAI model</li>
        <li>Enter your API key</li>
        <li>Ask questions about the data</li>
        <li>Get AI-powered insights and visualizations</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 20px 0; border-color: #333;'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: #9e9e9e; font-size: 0.8rem;'>Created with Streamlit & OpenAI</div>", unsafe_allow_html=True)

# Load selected dataset
if dataset_option == "Iris Flowers":
    df = load_iris()
    dataset_description_md = """
    ### Iris Flower Dataset
    
    This famous dataset contains measurements for 150 iris flowers from three different species:
    
    * Sepal length
    * Sepal width
    * Petal length
    * Petal width
    * Species (setosa, versicolor, virginica)
    """
elif dataset_option == "California Housing":
    df = load_boston_housing()
    dataset_description_md = """
    ### California Housing Dataset
    
    This dataset contains information about houses in California:
    
    * MedInc: median income in block group
    * HouseAge: median house age in block group
    * AveRooms: average number of rooms per household
    * AveBedrms: average number of bedrooms per household
    * Population: block group population
    * AveOccup: average number of household members
    * Latitude: block group latitude
    * Longitude: block group longitude
    * PRICE: median house value
    """
else:
    df = load_titanic()
    dataset_description_md = """
    ### Titanic Dataset
    
    This dataset contains information about passengers on the Titanic:
    
    * PassengerId: Unique ID for each passenger
    * Survived: Whether the passenger survived (0 = No, 1 = Yes)
    * Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
    * Name: Passenger name
    * Sex: Passenger sex
    * Age: Passenger age
    * SibSp: Number of siblings/spouses aboard
    * Parch: Number of parents/children aboard
    * Fare: Passenger fare
    """

# Main layout
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown('<div class="section-header">📋 Dataset Information</div>', unsafe_allow_html=True)
    
    # Use Streamlit's native markdown instead of HTML
    st.markdown(dataset_description_md)
    
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df, height=300, use_container_width=True)
    
    # Dataset overview metrics
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    # Use Streamlit columns for metrics instead of custom HTML
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with metrics_col2:
        st.metric("Columns", df.shape[1])
    with metrics_col3:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(1)
        st.metric("Missing Values", f"{missing_pct}%")
    
    # Simple dataset statistics
    st.markdown('<div class="section-header">📈 Quick Statistics</div>', unsafe_allow_html=True)
    
    # Numerical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if num_cols:
        with st.expander("📉 Numerical Features", expanded=True):
            st.dataframe(df[num_cols].describe().round(2), height=200, use_container_width=True)
    
                    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        with st.expander("📊 Categorical Features", expanded=True):
            for col in cat_cols:
                st.markdown(f"<b>{col}</b> distribution:", unsafe_allow_html=True)
                value_counts = df[col].value_counts().reset_index().rename(columns={'index': col, col: 'count'})
                st.dataframe(value_counts, height=100, use_container_width=True)
                
                # Add a small bar chart for categorical distributions
                if len(value_counts) <= 10:  # Only show chart if reasonable number of categories
                    fig, ax = plt.subplots(figsize=(6, 3))
                    # Use matplotlib's default style instead of custom dark mode
                    plt.style.use('default')
                    # Use a color that will be visible on both light and dark backgrounds
                    bars = ax.bar(value_counts[col].astype(str), value_counts['count'], color='#1976D2')
                    ax.set_ylabel('Count', color='black')
                    ax.set_title(f'{col} Distribution', color='black')
                    # Set tick parameters properly without 'ha' parameter
                    ax.tick_params(axis='x', colors='black')
                    ax.tick_params(axis='y', colors='black')
                    # Set rotation separately
                    plt.xticks(rotation=45)
                    # Adjust horizontal alignment by adjusting figure after ticks are set
                    plt.tight_layout()
                    # Ensure spines are visible
                    ax.spines['bottom'].set_color('black')
                    ax.spines['top'].set_color('black')
                    ax.spines['left'].set_color('black')
                    ax.spines['right'].set_color('black')
                    # Add values on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.0f}', ha='center', va='bottom', color='black')
                    fig.patch.set_facecolor('white')
                    ax.set_facecolor('white')
                    
                    # Remove the container div and just use pyplot directly
                    st.pyplot(fig)

with col2:
    st.markdown('<div class="section-header">🤖 Ask Your Data</div>', unsafe_allow_html=True)
    st.info("Type your question below and our AI assistant will analyze the data for you!")
    
    # Example questions
    with st.expander("💡 Example Questions", expanded=True):
        example_questions = {
            "Iris Flowers": [
                "Show me the relationship between sepal length and sepal width for different species.",
                "Which features are most important for distinguishing species?",
                "What's the average petal width for each species?"
            ],
            "California Housing": [
                "What factors most strongly affect housing prices?",
                "Show me the relationship between median income and house prices.",
                "How does location correlate with housing prices?"
            ],
            "Titanic": [
                "What was the survival rate by passenger class and gender?",
                "Did age affect survival probability?",
                "What was the average fare by passenger class?"
            ]
        }
        
        # Function to update the textarea with clicked question
        def set_question(question):
            st.session_state.user_question = question
        
        # Create buttons for example questions with improved styling
        for i, question in enumerate(example_questions[dataset_option]):
            button_key = f"question_button_{i}"
            st.button(
                question, 
                key=button_key,
                on_click=set_question,
                args=(question,),
                use_container_width=True
            )
    
    # Text area with the chosen question from session state
    user_question = st.text_area(
        "✍️ What would you like to know about this data?",
        value=st.session_state.user_question,
        height=100,
        key="question_input"
    )
    
    analyze_col1, analyze_col2 = st.columns([4, 1])
    with analyze_col1:
        analyze_button = st.button("🔍 Analyze Data", use_container_width=True)
    with analyze_col2:
        # Clear button that also clears session state
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.user_question = ""
            st.experimental_rerun()
    
    if analyze_button and user_question:
        if not api_key:
            st.error("⚠️ Please enter an OpenAI API key in the sidebar to use the AI assistant.")
        else:
            with st.spinner(f"🧠 Analyzing data with {openai_model} and generating insights..."):
                # Create a prompt with dataframe info
                df_info = f"DataFrame info:\n{df.info(verbose=False, memory_usage=False, buf=io.StringIO())}\n"
                df_head = f"First 5 rows:\n{df.head().to_string()}\n"
                df_describe = f"DataFrame description:\n{df.describe().to_string()}\n"
                
                prompt = f"""
                You're a data analysis assistant. I have the following DataFrame:
                
                {df_info}
                {df_head}
                {df_describe}
                
                My question is: {user_question}
                
                Please provide:
                1. A clear answer to my question based on the data
                2. Python code using matplotlib or seaborn to visualize this information (if applicable)
                
                For the visualization code:
                - Use pandas, matplotlib.pyplot as plt, and seaborn as sns
                - Do not include plt.show() as this will be handled separately
                - Make sure the code is complete and can run independently with just the dataframe (named 'df')
                - Add appropriate labels, titles, and legends to make the visualization clear
                - Use a light background for plots with dark text to ensure readability
                - Set figure size to be at least (10, 6) for better visibility
                - Ensure all text (labels, titles, ticks, legend) is black or dark colored
                - For categorical plots, include labels on the bars when appropriate
                - Add grid lines for better readability
                - Use contrasting colors for different categories that work well on white backgrounds
                - For scatter plots of categorical data, use larger marker sizes (s=80+)
                - For any pairplot or complex visualization, ensure font sizes are increased

                Format your answer with clear explanations and code in python code blocks when needed.
                """
                
                llm_response = query_llm(prompt, api_key, openai_model)
                
                # Display the text response (excluding code blocks)
                text_response = re.sub(r'```python.*?```', '', llm_response, flags=re.DOTALL)
                text_response = re.sub(r'```.*?```', '', text_response, flags=re.DOTALL)  # Remove any other code blocks
                
                st.markdown('<div class="section-header">🧠 AI Insights</div>', unsafe_allow_html=True)
                st.write(text_response.strip())
                
                # Extract and execute visualization code if present
                code = extract_code(llm_response)
                if code:
                    st.markdown('<div class="section-header">📊 Visualization</div>', unsafe_allow_html=True)
                    
                    # Show the code in an expander
                    with st.expander("💻 View Code"):
                        st.code(code, language="python")
                    
                    # Execute the code and display the plot directly without container
                    result = execute_plot_code(code, df)
                    
                    # Check if result is an error message or a base64 image
                    if result.startswith('data:') or len(result) > 100:
                        st.image(f"data:image/png;base64,{result}", use_column_width=True)
                    else:
                        st.error(f"⚠️ Error executing the code: {result}")
                        
                # Add a download button for the results
                st.download_button(
                    label="📥 Download Results", 
                    data=f"## AI Analysis Results\n\n### Question\n{user_question}\n\n### Answer\n{text_response.strip()}\n\n### Visualization Code\n```python\n{code if code else 'No visualization code generated.'}\n```",
                    file_name="data_analysis_results.md",
                    mime="text/markdown"
                )

# Footer
st.markdown("---")
st.write("Made with ❤️ using Streamlit, Pandas, and OpenAI")
st.write("© 2025 Peter Mamdouh")
