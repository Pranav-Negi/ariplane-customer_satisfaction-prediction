import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load model artifacts
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
model = joblib.load("svm_flight.pkl")

# Page configuration
st.set_page_config(page_title="✈️ Flight Satisfaction Prediction", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .description {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        color: #ffffff;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E40AF;
        border-bottom: 2px solid #1E40AF;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(to right, #1E40AF, #3B82F6);
        color: white;
        font-size: 1.5rem;
        height: 70px;
        width: 100%;
        border-radius: 10px;
        border: none;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #1E3A8A, #2563EB);
        transform: scale(1.02);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .form-container {
        background-color: #1e2126;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .centered-button {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .rating-section {
        background-color: #1e2126;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #1E40AF;
    }
    .section-title {
        font-weight: bold;
        color: #1E40AF;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    .rating-label {
        font-weight: 500;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    .delay-input {
        background-color: #1e2126;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .stSlider>div>div>div>div {
        background-color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">✈️ Flight Satisfaction Prediction</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="description">Please provide passenger details and service ratings. The app will predict if the passenger is satisfied or not.</p>',
    unsafe_allow_html=True)

# Mapping for categorical features
gender_map = {"Female": 0, "Male": 1}
customer_map = {"Loyal Customer": 0, "Disloyal Customer": 1}
travel_map = {"Business travel": 0, "Personal Travel": 1}
class_map = {"Eco": 0, "Eco Plus": 1, "Business": 2}

# --- Input Form ---
with st.form("prediction_form"):

    # Passenger Info
    st.markdown('<div class="sub-header">👤 Passenger Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", list(gender_map.keys()))
    with col2:
        customer_type = st.selectbox("Customer Type", list(customer_map.keys()))
    with col3:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)

    # Travel Details
    st.markdown('<div class="sub-header">🧳 Travel Details</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)
    with col4:
        travel_type = st.selectbox("Type of Travel", list(travel_map.keys()))
    with col5:
        travel_class = st.selectbox("Class", list(class_map.keys()))
    with col6:
        flight_distance = st.number_input("Flight Distance (miles)", min_value=0, value=500)

    # Service Ratings
    st.markdown('<div class="sub-header">⭐ Service Ratings (0–5)</div>', unsafe_allow_html=True)

    # Comfort & Entertainment
    st.markdown('<div class="section-title">Comfort & Entertainment</div>', unsafe_allow_html=True)
    col7, col8, col9 = st.columns(3)
    with col7:
        st.markdown('<p class="rating-label">Seat comfort</p>', unsafe_allow_html=True)
        seat_comfort = st.slider("Seat comfort", 1, 5, 3, label_visibility="collapsed")
        st.markdown('<p class="rating-label">Departure/Arrival time convenient</p>', unsafe_allow_html=True)
        dep_arr_conv = st.slider("Departure/Arrival time convenient", 1, 5, 3, label_visibility="collapsed")
    with col8:
        st.markdown('<p class="rating-label">Inflight wifi service</p>', unsafe_allow_html=True)
        inflight_wifi = st.slider("Inflight wifi service", 1, 5, 3, label_visibility="collapsed")
        st.markdown('<p class="rating-label">Inflight entertainment</p>', unsafe_allow_html=True)
        inflight_ent = st.slider("Inflight entertainment", 1, 5, 3, label_visibility="collapsed")
    with col9:
        st.markdown('<p class="rating-label">Leg room service</p>', unsafe_allow_html=True)
        legroom = st.slider("Leg room service", 1, 5, 3, label_visibility="collapsed")
        st.markdown('<p class="rating-label">Cleanliness</p>', unsafe_allow_html=True)
        cleanliness = st.slider("Cleanliness", 1, 5, 3, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Services & Support
    st.markdown('<div class="section-title">Services & Support</div>', unsafe_allow_html=True)
    col10, col11, col12 = st.columns(3)
    with col10:
        st.markdown('<p class="rating-label">Food and drink</p>', unsafe_allow_html=True)
        food_drink = st.slider("Food and drink", 1, 5, 3, label_visibility="collapsed")
        st.markdown('<p class="rating-label">Gate location</p>', unsafe_allow_html=True)
        gate_location = st.slider("Gate location", 1, 5, 3, label_visibility="collapsed")
    with col11:
        st.markdown('<p class="rating-label">Online support</p>', unsafe_allow_html=True)
        online_support = st.slider("Online support", 1, 5, 3, label_visibility="collapsed")
        st.markdown('<p class="rating-label">Ease of Online booking</p>', unsafe_allow_html=True)
        ease_booking = st.slider("Ease of Online booking", 1, 5, 3, label_visibility="collapsed")
    with col12:
        st.markdown('<p class="rating-label">On-board service</p>', unsafe_allow_html=True)
        onboard_service = st.slider("On-board service", 1, 5, 3, label_visibility="collapsed")
        st.markdown('<p class="rating-label">Online boarding</p>', unsafe_allow_html=True)
        online_boarding = st.slider("Online boarding", 1, 5, 3, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Baggage & Check-in
    st.markdown('<div class="section-title">Baggage & Check-in</div>', unsafe_allow_html=True)
    col13, col14 = st.columns(2)
    with col13:
        st.markdown('<p class="rating-label">Baggage handling</p>', unsafe_allow_html=True)
        baggage = st.slider("Baggage handling", 1, 5, 3, label_visibility="collapsed")
    with col14:
        st.markdown('<p class="rating-label">Check-in service</p>', unsafe_allow_html=True)
        checkin = st.slider("Check-in service", 1, 5, 3, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Flight Delays
    st.markdown('<div class="sub-header">⏱️ Flight Delays</div>', unsafe_allow_html=True)
    col15, col16 = st.columns(2)
    with col15:
        dep_delay = st.number_input("Departure Delay (Minutes)", min_value=0, value=0)
    with col16:
        arr_delay = st.number_input("Arrival Delay (Minutes)", min_value=0, value=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Centered button with custom CSS
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    submitted = st.form_submit_button("✨ Predict Satisfaction")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Logic ---
# --- Prediction Logic ---
if submitted:
    # Map categorical inputs back to numbers
    input_data = pd.DataFrame([[
        gender_map[gender], customer_map[customer_type], age,
        travel_map[travel_type], class_map[travel_class], flight_distance,
        seat_comfort, dep_arr_conv, food_drink, gate_location,
        inflight_wifi, inflight_ent, online_support, ease_booking,
        onboard_service, legroom, baggage, checkin, cleanliness,
        online_boarding, dep_delay, arr_delay
    ]], columns=columns)

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Save result in session_state
    st.session_state["prediction"] = prediction

# --- Show prediction result ---
if "prediction" in st.session_state:
    if st.session_state["prediction"] == 1:
        st.success("✅ Passenger is Satisfied")
    else:
        st.error("❌ Passenger is Not Satisfied")
