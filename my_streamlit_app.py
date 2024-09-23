import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("students_adaptability_level_online_education.csv")
df.dropna(inplace=True)

# Define target and features
target = 'Adaptivity Level'
features = df.drop(columns=[target])

# Encode categorical features
categorical_columns = ['Gender', 'Age', 'Education Level', 'Institution Type', 
                       'IT Student', 'Location', 'Load-shedding', 
                       'Financial Condition', 'Internet Type', 
                       'Network Type', 'Class Duration', 
                       'Self Lms', 'Device']
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    features[column] = label_encoders[column].fit_transform(features[column])

# Define X and encode target variable
X = features
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df[target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the model and label encoders
with open('adaptivity_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)
with open('target_encoder.pkl', 'wb') as file:
    pickle.dump(target_encoder, file)

# Streamlit app
st.set_page_config(page_title="Adaptability Level Prediction App", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff; /* Light background color */
    }
    .title {
        color: #007BFF; /* Title color */
    }
    .sidebar {
        background-color: #ffffff; /* Sidebar color */
        color: #333; /* Sidebar text color */
    }
    .result {
        font-weight: bold;
    }
    .low {
        color: red; /* Low adaptivity level */
    }
    .moderate {
        color: blue; /* Moderate adaptivity level */
    }
    .high {
        color: green; /* High adaptivity level */
    }
    .info-button {
        background-color: #007BFF; /* Button color */
        color: white; /* Text color */
        border-radius: 5px; /* Rounded corners */
        padding: 10px; /* Padding */
        font-weight: bold; /* Bold text */
        text-align: center; /* Centered text */
    }
    </style>
    """, unsafe_allow_html=True
)

st.title('📚 Student Adaptability Level Prediction App')
st.markdown(
    """
    This application predicts the adaptability level of students in online education based on various factors.
    Please fill in the details below to get a prediction. 🎓
    """
)

# Contact Info Button
if st.button('ℹ️ About the Developer'):
    st.markdown(
        """
        ### Developer Information
        **Name:** Ghazi Lassoued  
        **Email:** [lassouedghazi21@gmail.com](mailto:lassouedghazi21@gmail.com)  
        **Phone:** +21695292668  
        **LinkedIn:** [Ghazi Lassoued](https://www.linkedin.com/in/ghazi-lassoued-983419239/)  
        """
    )

# Sidebar for inputs
st.sidebar.header('🛠️ Input Student Data')

# Sidebar inputs with better organization and explanations
Gender = st.sidebar.selectbox('Gender 🚻', df['Gender'].unique())
Age = st.sidebar.selectbox('Age 🎂', df['Age'].unique())
Education_Level = st.sidebar.selectbox('Education Level 📘', df['Education Level'].unique())
Institution_Type = st.sidebar.selectbox('Institution Type 🏫', df['Institution Type'].unique())
IT_Student = st.sidebar.selectbox('IT Student 💻', df['IT Student'].unique())
Location = st.sidebar.selectbox('Location 🌍', df['Location'].unique())
Load_shedding = st.sidebar.selectbox('Load-shedding ⚡', df['Load-shedding'].unique())
Financial_Condition = st.sidebar.selectbox('Financial Condition 💰', df['Financial Condition'].unique())
Internet_Type = st.sidebar.selectbox('Internet Type 🌐', df['Internet Type'].unique())
Network_Type = st.sidebar.selectbox('Network Type 📶', df['Network Type'].unique())
Class_Duration = st.sidebar.selectbox('Class Duration ⏰', df['Class Duration'].unique())
Self_Lms = st.sidebar.selectbox('Self Lms 📚', df['Self Lms'].unique())
Device = st.sidebar.selectbox('Device 📱💻', df['Device'].unique())

if st.sidebar.button('🧙‍♂️ Predict Adaptability Level'):
    # Create the input_data DataFrame
    input_data = pd.DataFrame([{
        'Gender': Gender,
        'Age': Age,
        'Education Level': Education_Level,
        'Institution Type': Institution_Type,
        'IT Student': IT_Student,
        'Location': Location,
        'Load-shedding': Load_shedding,
        'Financial Condition': Financial_Condition,
        'Internet Type': Internet_Type,
        'Network Type': Network_Type,
        'Class Duration': Class_Duration,
        'Self Lms': Self_Lms,
        'Device': Device
    }])

    # Apply the same label encoding to the input data
    for column in categorical_columns:
        input_data[column] = label_encoders[column].transform(input_data[column])

    # Load the model and the target encoder, then make a prediction
    with open('adaptivity_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('target_encoder.pkl', 'rb') as file:
        target_encoder = pickle.load(file)
    prediction = model.predict(input_data)

    # Map predictions to actual category names
    adaptivity_level = target_encoder.inverse_transform(prediction)

    # Display the prediction result with color coding
    st.subheader('🎉 Prediction Result:')
    result_text = adaptivity_level[0]  # Get the actual label
    color_class = 'low' if result_text == 'Low' else 'moderate' if result_text == 'Moderate' else 'high'
    st.markdown(f'<p class="result {color_class}">{result_text} 📈</p>', unsafe_allow_html=True)

# Visualization Section
st.subheader('📊 Data Visualizations')
column_to_plot = st.selectbox('Select Column for Pie Chart 📊', df.columns)

if st.button('Show Pie Chart'):
    # Plotting the pie chart for the selected column
    if column_to_plot in df.columns:
        counts = df[column_to_plot].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.write(f"### {column_to_plot} Distribution")
        st.pyplot(fig)
    else:
        st.write("⚠️ Invalid column selected!")

# Relationship Table Section
if st.button('🔍 Show Relationship Table'):
    relationship_info = {
        'Column': [],
        'Description': [],
        'Relation with Target': []
    }
    
    for col in df.columns:
        if col != target:
            description = ""
            relation = ""

            if col == 'Gender':
                description = "The gender of the student (Male/Female)."
                relation = "Gender may influence adaptability levels due to different social and cultural factors."

            elif col == 'Age':
                description = "Age of the student."
                relation = "Different age groups may have varying adaptability levels in online education."

            elif col == 'Education Level':
                description = "The highest level of education attained."
                relation = "Higher education levels may correlate with better adaptability to online learning."

            elif col == 'Institution Type':
                description = "Type of institution (Public/Private)."
                relation = "The institution type may affect the resources available for online education."

            elif col == 'IT Student':
                description = "Whether the student is enrolled in IT-related courses (Yes/No)."
                relation = "IT students may have better technological skills, impacting adaptability."

            elif col == 'Location':
                description = "The geographical location of the student."
                relation = "Location can affect internet access and educational resources."

            elif col == 'Load-shedding':
                description = "Frequency of load-shedding in the student's area."
                relation = "Load-shedding can disrupt online education, affecting adaptability."

            elif col == 'Financial Condition':
                description = "Financial status of the student's household."
                relation = "Financial stability may influence access to educational resources and technology."

            elif col == 'Internet Type':
                description = "Type of internet connection available (e.g., DSL, Fiber)."
                relation = "Better internet types can enhance the online learning experience."

            elif col == 'Network Type':
                description = "Type of network used for internet access (e.g., 4G, 5G)."
                relation = "Higher network types generally provide better connectivity for online education."

            elif col == 'Class Duration':
                description = "Duration of online classes."
                relation = "Longer class durations may affect student engagement and adaptability."

            elif col == 'Self Lms':
                description = "Whether the student uses self-learning management systems (Yes/No)."
                relation = "Use of self-LMS may enhance adaptability by providing more learning resources."

            elif col == 'Device':
                description = "The device used for online classes (e.g., Laptop, Smartphone)."
                relation = "The type of device can impact the learning experience and adaptability."

            relationship_info['Column'].append(col)
            relationship_info['Description'].append(description)
            relationship_info['Relation with Target'].append(relation)

    relationship_df = pd.DataFrame(relationship_info)
    st.subheader('📈 Relationship of Features with Target Variable')
    st.write(relationship_df)

# Run the app with: streamlit run app.py

