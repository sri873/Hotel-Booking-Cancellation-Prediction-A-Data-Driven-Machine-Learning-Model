import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


st.title("\U0001F3E8 Hotel Booking Cancellation Predictor")

# Load dataset ONCE
@st.cache_data
def load_data():
    df = pd.read_csv("hotel_bookings.csv")
    df.drop(columns=['reservation_status', 'reservation_status_date'], inplace=True)

    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

data = load_data()

# Encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

# Balance dataset
df_canceled = data[data["is_canceled"] == 1].sample(40000, random_state=42)
df_not_canceled = data[data["is_canceled"] == 0].sample(40000, random_state=42)
df1 = pd.concat([df_canceled, df_not_canceled])
df1_indexes = df1.index.tolist()
training_data = df1.sample(frac=1, random_state=42).reset_index(drop=True)
testing_data = data.drop(df1_indexes).reset_index(drop=True)
testing_data = testing_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and target
X_train = training_data.drop(columns=['is_canceled'])
y_train = training_data['is_canceled']
X_test = testing_data.drop(columns=['is_canceled'])
y_test = testing_data['is_canceled']

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=1.0),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# Sidebar manual input form
st.sidebar.header("Enter Booking Details for Prediction")
with st.sidebar.form("prediction_form"):
    input_data = {
        'hotel': st.selectbox("Hotel", le_dict['hotel'].classes_),
        'lead_time': st.number_input("Lead Time", min_value=0, max_value=500),
        'arrival_date_year': st.selectbox("Arrival Year", list(range(2015, 2025))),
        'arrival_date_month': st.selectbox("Arrival Month", le_dict['arrival_date_month'].classes_),
        'arrival_date_week_number': st.number_input("Week Number", 1, 53),
        'arrival_date_day_of_month': st.number_input("Day of Month", 1, 31),
        'stays_in_weekend_nights': st.number_input("Weekend Nights", 0, 20),
        'stays_in_week_nights': st.number_input("Week Nights", 0, 50),
        'adults': st.number_input("Adults", 1, 5),
        'children': st.number_input("Children", 0, 5),
        'babies': st.number_input("Babies", 0, 5),
        'meal': st.selectbox("Meal", le_dict['meal'].classes_),
        'country': st.selectbox("Country", le_dict['country'].classes_),
        'market_segment': st.selectbox("Market Segment", le_dict['market_segment'].classes_),
        'distribution_channel': st.selectbox("Distribution Channel", le_dict['distribution_channel'].classes_),
        'is_repeated_guest': st.selectbox("Repeated Guest", [0, 1]),
        'previous_cancellations': st.number_input("Previous Cancellations", 0, 10),
        'previous_bookings_not_canceled': st.number_input("Previous Non-Cancellations", 0, 20),
        'reserved_room_type': st.selectbox("Reserved Room", le_dict['reserved_room_type'].classes_),
        'assigned_room_type': st.selectbox("Assigned Room", le_dict['assigned_room_type'].classes_),
        'booking_changes': st.number_input("Booking Changes", 0, 10),
        'deposit_type': st.selectbox("Deposit Type", le_dict['deposit_type'].classes_),
        'agent': st.number_input("Agent", 0, 600),
        'company': st.number_input("Company", 0, 600),
        'days_in_waiting_list': st.number_input("Days in Waiting List", 0, 100),
        'customer_type': st.selectbox("Customer Type", le_dict['customer_type'].classes_),
        'adr': st.number_input("ADR", 0.0, 1000.0),
        'required_car_parking_spaces': st.number_input("Parking Spaces", 0, 5),
        'total_of_special_requests': st.number_input("Special Requests", 0, 5)
    }
    submitted = st.form_submit_button("Predict")

# Select model
model_choice = st.selectbox("Choose Model to Train", list(models.keys()))
# Variable to store the trained model
trained_model = None
# TRAIN MODEL button
if st.button("Train Model"):
    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save the trained model to session state
    st.session_state.trained_model = model

    acc = accuracy_score(y_test, y_pred)
    st.success(f"{model_choice} Accuracy: {acc:.4f}")

    # Show classification report as a table
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.subheader("📋 Classification Report")
    st.dataframe(report_df.style.format({
        "precision": "{:.2f}", "recall": "{:.2f}",
        "f1-score": "{:.2f}", "support": "{:.0f}"
    }))

# PREDICT BUTTON from sidebar
if submitted:
    if 'trained_model' in st.session_state:  # Check if the model is trained
        trained_model = st.session_state.trained_model
   
        manual_df = pd.DataFrame([input_data])
        for col in manual_df.select_dtypes(include='object').columns:
            manual_df[col] = le_dict[col].transform(manual_df[col])
        scaled_input = scaler.transform(manual_df)
        pred = trained_model.predict(scaled_input)[0]
        label = "Will be Cancelled ❌" if pred == 1 else "Will NOT be Cancelled ✅"
        st.success(f"🔮 Prediction for Manual Input: {label}")
    else:
        st.warning("⚠️ Please train the model first before predicting!")
# 


# COMPARE MODELS button
if st.button("Compare Models"):
    all_acc = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred_all = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred_all)
        all_acc[name] = acc

    acc_df = pd.DataFrame.from_dict(all_acc, orient='index', columns=['Accuracy'])
    acc_df = acc_df.sort_values(by='Accuracy', ascending=False)
    
    st.subheader("📊 Model Accuracy Comparison")
    st.bar_chart(acc_df)

    # Correlation plot (top 10)
    st.subheader(f"🔍 Top 10 Features by Correlation with Target")
    corr_vals = data.corr()['is_canceled'].drop('is_canceled').abs().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=corr_vals.values, y=corr_vals.index, ax=ax)
    ax.set_title('Feature Correlations with is_canceled')
    st.pyplot(fig)
