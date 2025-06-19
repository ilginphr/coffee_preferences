import streamlit as st
from coffe_model import train_model
model, le_mood, le_tol, le_pref, df = train_model()

st.title("☕ Coffee Preference App")
st.markdown("☕ Tell us a bit about yourself and we'll guess your perfect cup of coffee! 🚀")

age = st.slider("Age", 18, 60, 25)
sleep = st.slider("How many hours do you usually sleep?", 4, 14, 7)

mood_options = {
    "😴 Tired": "Tired",
    "😐 Neutral": "Neutral",
    "⚡ Energetic": "Energetic"
}
mood_choice = st.selectbox("Your current mood:", list(mood_options.keys()))
mood_str = mood_options[mood_choice]  # 'Tired', 'Neutral' veya 'Energetic'

tolerance_options = {
    "🥱 Low": "Low",
    "🙂 Medium": "Medium",
    "🚀 High": "High"
}
tolerance_choice = st.selectbox("Caffeine tolerance level:", list(tolerance_options.keys()))
tolerance_str = tolerance_options[tolerance_choice]  # 'Low', 'Medium' veya 'High'

# Encode
mood = le_mood.transform([mood_str])[0]
tolerance = le_tol.transform([tolerance_str])[0]

#st.write("Mood code:", mood)
#st.write("Tolerans code:", tolerance)


if st.button("Guess my coffee!"):
    input_data = [[age, sleep, mood, tolerance]]
    prediction = model.predict(input_data)[0]
    predicted_label = le_pref.inverse_transform([prediction])[0]

    st.success(f"🎯 Your ideal coffee is: **{predicted_label}**")
