# Talk to Your Data

A Streamlit application that allows users to analyze data through natural language questions using AI assistance.

![Talk to Your Data Demo](demo_screenshot.png)

## Features

- 📊 Interactive data analysis using natural language
- 🤖 AI-powered insights using OpenAI models
- 📈 Automatic visualization generation
- 🔍 Built-in example datasets (Iris, California Housing, Titanic)
- 💬 Example questions to guide users
- 📱 Responsive dark-themed UI

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/talk-to-your-data.git
cd talk-to-your-data
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Select a dataset from the sidebar dropdown
2. Enter your OpenAI API key
3. Choose an OpenAI model (GPT-3.5-turbo, GPT-4, etc.)
4. Type a question about the data or click on an example question
5. Click "Analyze Data" to get AI-generated insights and visualizations

## Example Questions

### Iris Dataset
- Show me the relationship between sepal length and sepal width for different species
- Which features are most important for distinguishing species?
- What's the average petal width for each species?

### California Housing Dataset
- What factors most strongly affect housing prices?
- Show me the relationship between median income and house prices
- How does location correlate with housing prices?

### Titanic Dataset
- What was the survival rate by passenger class and gender?
- Did age affect survival probability?
- What was the average fare by passenger class?

## Components

### Data Loading
- Uses sample datasets (Iris, California Housing, Titanic)
- Data loading functions are cached for performance

### UI Components
- Dark-themed responsive interface
- Two-column layout for information and interaction
- Expandable sections for dataset details

### Data Analysis
- OpenAI API integration for natural language processing
- Automatic extraction and execution of Python code
- Dynamic visualization generation

### Visualization
- Custom styling for visibility on dark backgrounds
- Proper handling of different chart types
- Value annotations on categorical charts

## Configuration Options

### Style Settings
- Light plots with dark text for better readability
- Configurable plot size and resolution
- Clear axis labels and grid lines

### OpenAI Models
- Support for GPT-3.5-turbo, GPT-4, GPT-4-turbo, and GPT-4o
- Customizable prompts for better analysis

## Project Structure

```
talk-to-your-data/
├── app.py                  # Main application file
├── requirements.txt        # Package dependencies
├── README.md               # Project documentation
└── assets/                 # Static assets (if any)
```

## Dependencies

```
streamlit>=1.25.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0         # For sample datasets
openai>=1.0.0               # For API integration
numpy>=1.24.0
```

## Customization

### Adding Custom Datasets
You can add your own datasets by creating new loading functions:

```python
@st.cache_data
def load_your_dataset():
    # Load and return your dataframe
    df = pd.read_csv("your_file.csv")
    return df
```

### Styling Modifications
Adjust the visualization style by modifying the plot parameters:

```python
plt.rcParams['figure.figsize'] = (12, 8)  # Change figure size
plt.rcParams['axes.labelsize'] = 14       # Change label size
```

## License

MIT License

## Credits

- Developed by [Your Name]
- Built with Streamlit and OpenAI
- Visualization using Matplotlib and Seaborn
