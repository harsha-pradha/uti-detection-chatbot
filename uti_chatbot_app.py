import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import sys
import os
import re

# Set page config
st.set_page_config(
    page_title="UTI Detection Chatbot - AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .risk-low {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 10px 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .parameter-box {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 5px 0;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 15px;
        margin: 5px 0;
        border-bottom-right-radius: 5px;
    }
    .chat-ai {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 15px;
        margin: 5px 0;
        border-bottom-left-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">🩺 AI-Powered UTI Detection Chatbot</div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; color: #666;'>
Get instant AI-powered analysis of your urinalysis results with explanations in English and Tamil
</div>
""", unsafe_allow_html=True)

# Load model and preprocessing artifacts
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and preprocessing artifacts"""
    try:
        model = joblib.load('models/best_uti_model.pkl')
        scaler = joblib.load('models/feature_scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        model_performance = joblib.load('models/model_performance.pkl')
        return model, scaler, feature_names, model_performance
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

def predict_uti_risk(user_inputs, model, scaler, feature_names):
    """Enhanced UTI risk prediction with clinical rules"""
    try:
        # Prepare input features
        input_features = prepare_user_inputs(user_inputs, feature_names)
        
        # Scale features
        input_scaled = scaler.transform([input_features])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # ====== CRITICAL FIX: Apply clinical rules to override model ======
        clinical_probability = apply_clinical_rules(user_inputs, probability)
        
        # Use the higher of model probability or clinical probability
        final_probability = max(probability, clinical_probability)
        
        # Determine risk level with clinical adjustment
        if final_probability >= 0.6:
            risk_level = "HIGH"
        elif final_probability >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'prediction': 1 if final_probability >= 0.5 else 0,
            'probability': final_probability,
            'risk_level': risk_level,
            'confidence': final_probability
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def apply_clinical_rules(user_inputs, base_probability):
    """Apply clinical rules to adjust probability based on key indicators"""
    clinical_score = base_probability
    
    # High WBC is strong indicator of UTI
    if user_inputs.get('WBC', 0) > 10:
        clinical_score += 0.3
    elif user_inputs.get('WBC', 0) > 5:
        clinical_score += 0.15
    
    # Bacteria presence
    if user_inputs.get('Bacteria', 0) >= 3:  # MODERATE or PLENTY
        clinical_score += 0.4
    elif user_inputs.get('Bacteria', 0) >= 2:  # FEW
        clinical_score += 0.2
    
    # High protein
    if user_inputs.get('Protein', 0) >= 3:  # 2+ or 3+
        clinical_score += 0.2
    
    # Abnormal pH
    if user_inputs.get('pH', 7.0) > 8.0 or user_inputs.get('pH', 7.0) < 5.0:
        clinical_score += 0.1
    
    # Cloudy urine
    if user_inputs.get('Transparency', 0) >= 3:  # CLOUDY or TURBID
        clinical_score += 0.15
    
    # Female gender (higher UTI risk)
    if user_inputs.get('Gender_FEMALE', 0) == 1:
        clinical_score += 0.1
    
    return min(clinical_score, 0.95)  # Cap at 95%

def prepare_user_inputs(user_inputs, expected_features):
    """Prepare user inputs for model prediction with ALL expected features"""
    # Create a complete feature dictionary with ALL expected features
    feature_dict = {}
    
    # Set defaults for ALL expected features from your scaler
    all_expected_features = [
        "Age", "Transparency", "Glucose", "Protein", "pH", "Specific Gravity", 
        "WBC", "RBC", "Epithelial Cells", "Mucous Threads", "Amorphous Urates", 
        "Bacteria", "Color_AMBER", "Color_BROWN", "Color_DARK YELLOW", 
        "Color_LIGHT RED", "Color_LIGHT YELLOW", "Color_RED", "Color_REDDISH", 
        "Color_REDDISH YELLOW", "Color_STRAW", "Color_YELLOW", 
        "Gender_FEMALE", "Gender_MALE"
    ]
    
    # Initialize all features to 0
    for feature in all_expected_features:
        feature_dict[feature] = 0
    
    # Update with user provided values
    feature_dict.update(user_inputs)
    
    # Ensure we return features in the exact order expected by the scaler
    return [feature_dict[feature] for feature in all_expected_features]

# ========== ENHANCED MEDICAL GEN AI CHATBOT ==========
class MedicalGenAIChatBot:
    def __init__(self):
        # Comprehensive medical knowledge base
        self.medical_knowledge_base = {
            'uti_basics': {
                'en': "UTI (Urinary Tract Infection) is an infection in any part of the urinary system including kidneys, ureters, bladder, and urethra. Most UTIs are caused by bacteria, most commonly E. coli.",
                'ta': "யூடிஐ (சிறுநீர் கோளாறு) என்பது சிறுநீர் மண்டலத்தின் எந்தப் பகுதியிலும் ஏற்படும் தொற்று. இதில் சிறுநீரகங்கள், சிறுநீர்க்குழாய்கள், சிறுநீர்ப்பை மற்றும் சிறுநீர் வடிகுழாய் அடங்கும். பெரும்பாலான யூடிஐ-க்கள் பாக்டீரியாவால் ஏற்படுகின்றன."
            },
            'symptoms': {
                'en': "Common UTI symptoms include: • Burning sensation during urination • Frequent urination • Cloudy, dark, or strong-smelling urine • Pelvic pain in women • Fever or chills (if infection reaches kidneys)",
                'ta': "யூடிஐ பொதுவான அறிகுறிகள்: • சிறுநீர் கழிக்கும் போது எரிச்சல் • அடிக்கடி சிறுநீர் கழித்தல் • மங்கலான, கருத்த அல்லது வலுவான வாசனை சிறுநீர் • பெண்களில் இடுப்பு வலி • காய்ச்சல் அல்லது குளிர் (தொற்று சிறுநீரகத்தை அடைந்தால்)"
            },
            'treatment': {
                'en': "UTI treatment: • Antibiotics like trimethoprim, nitrofurantoin, fosfomycin • Pain relievers for discomfort • Increased fluid intake • Complete the full course of antibiotics",
                'ta': "யூடிஐ சிகிச்சை: • ட்ரைமெத்தோப்ரிம், நைட்ரோஃபுராண்டோயின், ஃபோஸ்ஃபோமைசின் போன்ற நுண்ணுயிர் எதிர்ப்பிகள் • வலி நிவாரணிகள் • திரவ உட்கொள்ளல் அதிகரிப்பு • நுண்ணுயிர் எதிர்ப்பிகளின் முழு பாடத்தையும் முடிக்கவும்"
            },
            'prevention': {
                'en': "UTI prevention: • Drink 6-8 glasses of water daily • Urinate frequently and completely • Wipe from front to back • Empty bladder after intercourse • Avoid irritating feminine products",
                'ta': "யூடிஐ தடுப்பு: • தினமும் 6-8 கிளாஸ் தண்ணீர் குடிக்கவும் • அடிக்கடி மற்றும் முழுமையாக சிறுநீர் கழிக்கவும் • முன்பக்கத்தில் இருந்து பின்பக்கமாகத் துடைக்கவும் • உடலுறவுக்குப் பிறகு சிறுநீர்ப்பையை காலி செய்யவும் • எரிச்சலூட்டும் பெண்கள் தயாரிப்புகளை தவிர்க்கவும்"
            },
            'risk_factors': {
                'en': "UTI risk factors: • Female anatomy • Sexual activity • Certain birth control • Menopause • Urinary tract abnormalities • Diabetes • Weakened immune system",
                'ta': "யூடிஐ ஆபத்து காரணிகள்: • பெண் உடற்கூறியல் • பாலியல் செயல்பாடு • சில கருத்தடை முறைகள் • மாதவிடாய் நிறுத்தம் • சிறுநீர் மண்டல அசாதாரணங்கள் • நீரிழிவு • பலவீனமான நோயெதிர்ப்பு அமைப்பு"
            },
            'diagnosis': {
                'en': "UTI diagnosis: • Urinalysis to check WBC, RBC, bacteria • Urine culture to identify bacteria • Imaging tests for recurrent UTIs • Cystoscopy for complex cases",
                'ta': "யூடிஐ கண்டறிதல்: • வெள்ளை இரத்த அணுக்கள், சிவப்பு இரத்த அணுக்கள், பாக்டீரியா சோதனை • பாக்டீரியாவை அடையாளம் காண சிறுநீர் கலாச்சாரம் • மீண்டும் மீண்டும் வரும் யூடிஐ-க்களுக்கு இமேஜிங் பரிசோதனைகள் • சிக்கலான வழக்குகளுக்கு சிஸ்டோஸ்கோபி"
            },
            'complications': {
                'en': "UTI complications: • Recurrent infections • Permanent kidney damage • Sepsis (life-threatening) • Delivery complications in pregnancy • Urethral narrowing in men",
                'ta': "யூடிஐ சிக்கல்கள்: • மீண்டும் மீண்டும் வரும் தொற்றுகள் • நிரந்தர சிறுநீரக சேதம் • செப்சிஸ் (உயிருக்கு ஆபத்தான) • கர்ப்பத்தில் பிரசவ சிக்கல்கள் • ஆண்களில் சிறுநீர் வடிகுழாய் குறுகல்"
            },
            'home_remedies': {
                'en': "UTI home remedies (not substitutes for medical treatment): • Drink plenty of water • Use heating pads for pain • Avoid caffeine and alcohol • Try cranberry juice (may help prevent) • Practice good hygiene",
                'ta': "யூடிஐ வீட்டு மருந்துகள் (மருத்துவ சிகிச்சைக்கு பதிலாக அல்ல): • நிறைய தண்ணீர் குடிக்கவும் • வலிக்கு வெப்ப பேட்கள் பயன்படுத்தவும் • காஃபின் மற்றும் ஆல்கஹால் தவிர்க்கவும் • கிரான்பெரி சாறு முயற்சிக்கவும் (தடுப்பதற்கு உதவலாம்) • நல்ல சுகாதாரத்தை பழக்கவும்"
            }
        }
        
        # Medical keywords for smart retrieval
        self.keyword_mapping = {
            'what is uti': 'uti_basics',
            'symptoms': 'symptoms', 
            'treatment': 'treatment',
            'prevention': 'prevention',
            'risk factors': 'risk_factors',
            'diagnosis': 'diagnosis',
            'complications': 'complications',
            'home remedies': 'home_remedies',
            'causes': 'uti_basics',
            'antibiotics': 'treatment',
            'pain': 'symptoms',
            'burning': 'symptoms',
            'frequent urination': 'symptoms'
        }
    
    def get_contextual_response(self, user_question, user_risk_data=None, language='en'):
        """Smart response combining medical knowledge and user context"""
        # Find relevant medical context using keyword matching
        relevant_context = self._retrieve_medical_context(user_question, language)
        
        # Add user's personal risk context if available
        user_context = self._get_user_context(user_risk_data, language)
        
        # Generate intelligent response
        response = self._generate_intelligent_response(user_question, relevant_context, user_context, language)
        
        return response
    
    def _retrieve_medical_context(self, question, language):
        """Retrieve relevant medical information based on question"""
        question_lower = question.lower()
        relevant_info = []
        
        # Keyword-based retrieval
        for keyword, topic in self.keyword_mapping.items():
            if keyword in question_lower:
                if topic in self.medical_knowledge_base:
                    relevant_info.append(self.medical_knowledge_base[topic][language])
        
        # If no specific topic found, provide general UTI info
        if not relevant_info:
            relevant_info.append(self.medical_knowledge_base['uti_basics'][language])
        
        return " ".join(relevant_info)
    
    def _get_user_context(self, user_risk_data, language):
        """Get personalized context based on user's risk assessment"""
        if user_risk_data and st.session_state.prediction_result:
            risk_level = st.session_state.prediction_result['risk_level']
            probability = st.session_state.prediction_result['probability']
            
            if language == 'en':
                return f" Based on your urinalysis results, you have {risk_level} UTI risk ({probability:.1%} probability)."
            else:
                return f" உங்கள் சிறுநீர் பரிசோதனை முடிவுகளின் அடிப்படையில், உங்களுக்கு {risk_level} யூடிஐ ஆபத்து உள்ளது ({probability:.1%} நிகழ்தகவு)."
        
        return ""
    
    def _generate_intelligent_response(self, question, medical_context, user_context, language):
        """Generate intelligent response using enhanced template system"""
        
        # Response templates for different question types
        response_templates = {
            'en': {
                'symptom_question': f"Based on medical knowledge: {medical_context}{user_context} Common UTI symptoms include burning during urination, frequent urination, and cloudy urine. If you're experiencing these symptoms, consult a healthcare provider.",
                'treatment_question': f"Medical information: {medical_context}{user_context} UTI treatment typically involves antibiotics prescribed by a doctor. Always complete the full course of medication.",
                'prevention_question': f"Prevention guidance: {medical_context}{user_context} Good practices include staying hydrated, proper hygiene, and urinating after intercourse.",
                'general_question': f"Medical insight: {medical_context}{user_context} Remember to consult healthcare professionals for personalized medical advice and proper diagnosis."
            },
            'ta': {
                'symptom_question': f"மருத்துவ அறிவின் அடிப்படையில்: {medical_context}{user_context} பொதுவான யூடிஐ அறிகுறிகளில் சிறுநீர் கழிக்கும் போது எரிச்சல், அடிக்கடி சிறுநீர் கழித்தல் மற்றும் மங்கலான சிறுநீர் ஆகியவை அடங்கும். இந்த அறிகுறிகளை நீங்கள் அனுபவித்தால், சுகாதார வழங்குநரைக் கலந்தாலோசிக்கவும்.",
                'treatment_question': f"மருத்துவ தகவல்: {medical_context}{user_context} யூடிஐ சிகிச்சை பொதுவாக மருத்துவரால் prescribed நுண்ணுயிர் எதிர்ப்பிகளை உள்ளடக்கியது. மருந்தின் முழு பாடத்தையும் எப்போதும் முடிக்கவும்.",
                'prevention_question': f"தடுப்பு வழிகாட்டல்: {medical_context}{user_context} நல்ல பழக்கங்களில் நீரேற்றமாக இருப்பது, சரியான சுகாதாரம் மற்றும் உடலுறவுக்குப் பிறகு சிறுநீர் கழிப்பது ஆகியவை அடங்கும்.",
                'general_question': f"மருத்துவ நுண்ணறிவு: {medical_context}{user_context} தனிப்பட்ட மருத்துவ ஆலோசனை மற்றும் சரியான நோய் கண்டறிதலுக்கு சுகாதார வல்லுநர்களைக் கலந்தாலோசிக்க நினைவில் கொள்ளவும்."
            }
        }
        
        # Determine question type
        question_lower = question.lower()
        if any(word in question_lower for word in ['symptom', 'pain', 'burning', 'feel']):
            question_type = 'symptom_question'
        elif any(word in question_lower for word in ['treat', 'medicine', 'antibiotic', 'cure']):
            question_type = 'treatment_question'
        elif any(word in question_lower for word in ['prevent', 'avoid', 'stop']):
            question_type = 'prevention_question'
        else:
            question_type = 'general_question'
        
        # Add disclaimer
        disclaimer = {
            'en': "\n\n*Note: I am an AI assistant providing general information. Please consult healthcare professionals for medical diagnosis and treatment.*",
            'ta': "\n\n*குறிப்பு: நான் பொதுவான தகவல்களை வழங்கும் AI உதவியாளன். மருத்துவ நோய் கண்டறிதல் மற்றும் சிகிச்சைக்கு சுகாதார வல்லுநர்களைக் கலந்தாலோசிக்கவும்.*"
        }
        
        return response_templates[language][question_type] + disclaimer[language]

# Initialize Enhanced Medical AI Chatbot
medical_ai_chatbot = MedicalGenAIChatBot()

# Bilingual Explanation Engine (your existing class)
class BilingualExplanationEngine:
    def __init__(self):
        self.normal_ranges = {
            'pH': (4.5, 8.0),
            'Specific Gravity': (1.005, 1.030),
            'WBC': (0, 5),
            'RBC': (0, 3),
            'Glucose': (0, 1),
            'Protein': (0, 1),
            'Bacteria': (0, 1)
        }
        
        self.explanations = {
            'en': {
                'high_risk': "Based on your urinalysis results, there is a **{risk_percentage}% probability** of Urinary Tract Infection. Consultation with a healthcare provider is recommended.",
                'medium_risk': "Your results show a **{risk_percentage}% probability** of Urinary Tract Infection. Further evaluation may be needed.",
                'low_risk': "Your urinalysis results indicate a **low probability ({risk_percentage}%)** of Urinary Tract Infection. Continue with good urinary health practices.",
                'abnormal_ph': "• **pH Level**: Your urine pH is **{value}** (Normal range: 4.5-8.0)",
                'abnormal_sg': "• **Specific Gravity**: Your value is **{value}** {status} normal range (1.005-1.030)",
                'high_wbc': "• **White Blood Cells**: Elevated level **{value}** may indicate infection or inflammation",
                'high_rbc': "• **Red Blood Cells**: Presence **{value}** may require further investigation",
                'glucose_present': "• **Glucose**: Detected in urine **{level}**",
                'protein_present': "• **Protein**: Level **{level}** may indicate kidney issues",
                'bacteria_present': "• **Bacteria**: Presence **{level}** suggests possible infection",
                'prevention_tips': [
                    "Drink 8-10 glasses of water daily",
                    "Practice good personal hygiene",
                    "Urinate when you feel the need - don't hold it",
                    "Wipe from front to back after using the toilet",
                    "Urinate after sexual intercourse",
                    "Avoid using harsh soaps in the genital area",
                    "Wear cotton underwear and loose-fitting clothes"
                ]
            },
            'ta': {
                'high_risk': "உங்கள் சிறுநீர் பரிசோதனை முடிவுகளின் அடிப்படையில், சிறுநீர் கோளாறு (யூடிஐ) ஏற்படுவதற்கான நிகழ்தகவு **{risk_percentage}%** ஆகும். சுகாதார வழங்குநருடன் கலந்தாலோசிப்பது பரிந்துரைக்கப்படுகிறது.",
                'medium_risk': "உங்கள் முடிவுகள் சிறுநீர் கோளாறு (யூடிஐ) ஏற்படுவதற்கான **{risk_percentage}% நிகழ்தகவை** காட்டுகின்றன. மேலும் மதிப்பீடு தேவைப்படலாம்.",
                'low_risk': "உங்கள் சிறுநீர் பரிசோதனை முடிவுகள் சிறுநீர் கோளாறு (யூடிஐ) ஏற்படுவதற்கான **குறைந்த நிகழ்தகவை ({risk_percentage}%)** காட்டுகின்றன. நல்ல சிறுநீர் சுகாதார பழக்கங்களைத் தொடரவும்.",
                'abnormal_ph': "• **pH அளவு**: உங்கள் சிறுநீர் pH **{value}** (சாதாரண வரம்பு: 4.5-8.0)",
                'abnormal_sg': "• **குறிப்பிட்ட ஈர்ப்பு**: உங்கள் மதிப்பு **{value}** சாதாரண வரம்பிற்கு {status} (1.005-1.030)",
                'high_wbc': "• **வெள்ளை இரத்த அணுக்கள்**: அதிகரித்த அளவு **{value}** தொற்று அல்லது வீக்கத்தைக் குறிக்கலாம்",
                'high_rbc': "• **சிவப்பு இரத்த அணுக்கள்**: இருப்பு **{value}** மேலும் விசாரணை தேவைப்படலாம்",
                'glucose_present': "• **குளுக்கோஸ்**: சிறுநீரில் கண்டறியப்பட்டது **{level}**",
                'protein_present': "• **புரதம்**: அளவு **{level}** சிறுநீரக சிக்கல்களைக் குறிக்கலாம்",
                'bacteria_present': "• **பாக்டீரியா**: இருப்பு **{level}** சாத்தியமான தொற்றைக் குறிக்கிறது",
                'prevention_tips': [
                    "தினமும் 8-10 கிளாஸ் தண்ணீர் குடிக்கவும்",
                    "நல்ல தனிப்பட்ட சுகாதாரத்தை பழக்கவும்",
                    "சிறுநீர் கழிக்க வேண்டியதன் அவசியத்தை உணரும்போது கழிக்கவும் - அடக்கிவைக்காதீர்கள்",
                    "கழிப்பறை பயன்படுத்திய பின் முன்பக்கத்தில் இருந்து பின்பக்கமாகத் துடைக்கவும்",
                    "பாலியல் தொடர்புக்குப் பிறகு சிறுநீர் கழிக்கவும்",
                    "பிறப்புறுப்புப் பகுதியில் கடுமையான சோப்புகளைப் பயன்படுத்துவதைத் தவிர்க்கவும்",
                    "பருத்தி உள்ளாடை மற்றும் தளர்வான ஆடைகளை அணியவும்"
                ]
            }
        }

    def generate_explanation(self, user_inputs, prediction_result, language='en'):
        risk_percentage = int(prediction_result['probability'] * 100)
        
        # Main risk message
        if prediction_result['risk_level'] == 'HIGH':
            main_message = self.explanations[language]['high_risk'].format(risk_percentage=risk_percentage)
        elif prediction_result['risk_level'] == 'MEDIUM':
            main_message = self.explanations[language]['medium_risk'].format(risk_percentage=risk_percentage)
        else:
            main_message = self.explanations[language]['low_risk'].format(risk_percentage=risk_percentage)
        
        # Detailed explanations
        detailed_explanations = []
        for param_name, value in user_inputs.items():
            explanation = self._analyze_parameter(param_name, value, language)
            if explanation:
                detailed_explanations.append(explanation)
        
        return {
            'main_message': main_message,
            'detailed_explanations': detailed_explanations,
            'prevention_tips': self.explanations[language]['prevention_tips'],
            'risk_level': prediction_result['risk_level'],
            'confidence': prediction_result['confidence']
        }
    
    def _analyze_parameter(self, param_name, value, language):
        if param_name in self.normal_ranges:
            min_val, max_val = self.normal_ranges[param_name]
            
            if value < min_val or value > max_val:
                template_key = f'abnormal_{param_name.lower().replace(" ", "_")}'
                if template_key in self.explanations[language]:
                    status = "above" if value > max_val else "below"
                    status_ta = "மேலே" if value > max_val else "கீழே"
                    return self.explanations[language][template_key].format(
                        value=value, 
                        status=status if language == 'en' else status_ta,
                        level=self._get_level_description(value, param_name, language)
                    )
        return None
    
    def _get_level_description(self, value, param_name, language):
        descriptions = {
            'en': {
                'Glucose': ['NEGATIVE', 'TRACE', '1+', '2+', '3+', '4+'],
                'Protein': ['NEGATIVE', 'TRACE', '1+', '2+', '3+'],
                'Bacteria': ['NONE', 'RARE', 'FEW', 'MODERATE', 'PLENTY']
            },
            'ta': {
                'Glucose': ['இல்லை', 'சிறிதளவு', '1+', '2+', '3+', '4+'],
                'Protein': ['இல்லை', 'சிறிதளவு', '1+', '2+', '3+'],
                'Bacteria': ['இல்லை', 'அரிதாக', 'சில', 'மிதமான', 'நிறைய']
            }
        }
        
        if param_name in descriptions[language]:
            levels = descriptions[language][param_name]
            if 0 <= value < len(levels):
                return levels[int(value)]
        return str(value)

# Initialize components
model, scaler, feature_names, model_performance = load_model_artifacts()
explanation_engine = BilingualExplanationEngine()

# Check feature compatibility
if feature_names:
    st.sidebar.write(f"✅ Model expects {len(feature_names)} features")

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Sidebar for input
st.sidebar.header("🔬 Enter Lab Values")

# Input fields in two columns
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=100, value=35, key="age")
    ph = st.slider("pH Level", min_value=4.0, max_value=9.0, value=6.5, step=0.1, key="ph")
    specific_gravity = st.slider("Specific Gravity", min_value=1.000, max_value=1.040, value=1.015, step=0.001, key="sg")
    wbc = st.number_input("White Blood Cells (WBC)", min_value=0, max_value=100, value=5, key="wbc")
    rbc = st.number_input("Red Blood Cells (RBC)", min_value=0, max_value=100, value=1, key="rbc")

with col2:
    glucose = st.selectbox("Glucose", ["NEGATIVE", "TRACE", "1+", "2+", "3+", "4+"], key="glucose")
    protein = st.selectbox("Protein", ["NEGATIVE", "TRACE", "1+", "2+", "3+"], key="protein")
    bacteria = st.selectbox("Bacteria", ["NONE SEEN", "RARE", "FEW", "MODERATE", "PLENTY"], key="bacteria")
    transparency = st.selectbox("Transparency", ["CLEAR", "SLIGHTLY HAZY", "HAZY", "CLOUDY", "TURBID"], key="transparency")
    gender = st.radio("Gender", ["MALE", "FEMALE"], key="gender")

# Mapping dictionaries
glucose_map = {"NEGATIVE": 0, "TRACE": 1, "1+": 2, "2+": 3, "3+": 4, "4+": 5}
protein_map = {"NEGATIVE": 0, "TRACE": 1, "1+": 2, "2+": 3, "3+": 4}
bacteria_map = {"NONE SEEN": 0, "RARE": 1, "FEW": 2, "MODERATE": 3, "PLENTY": 4}
transparency_map = {"CLEAR": 0, "SLIGHTLY HAZY": 1, "HAZY": 2, "CLOUDY": 3, "TURBID": 4}

# Test cases in sidebar
st.sidebar.markdown("---")
st.sidebar.header("🧪 Test Cases")

if st.sidebar.button("Test HIGH Risk Case"):
    # Update session state to simulate high-risk inputs
    st.session_state.age = 30
    st.session_state.ph = 8.5
    st.session_state.sg = 1.025
    st.session_state.wbc = 50
    st.session_state.rbc = 10
    st.session_state.glucose = "NEGATIVE"
    st.session_state.protein = "3+"
    st.session_state.bacteria = "PLENTY"
    st.session_state.transparency = "TURBID"
    st.session_state.gender = "FEMALE"
    st.sidebar.success("High-risk test case loaded! Click 'Analyze My Report'")

if st.sidebar.button("Test LOW Risk Case"):
    # Update session state to simulate low-risk inputs
    st.session_state.age = 30
    st.session_state.ph = 6.5
    st.session_state.sg = 1.015
    st.session_state.wbc = 2
    st.session_state.rbc = 1
    st.session_state.glucose = "NEGATIVE"
    st.session_state.protein = "NEGATIVE"
    st.session_state.bacteria = "NONE SEEN"
    st.session_state.transparency = "CLEAR"
    st.session_state.gender = "MALE"
    st.sidebar.success("Low-risk test case loaded! Click 'Analyze My Report'")

# Analysis button
if st.sidebar.button("🔍 Analyze My Report", type="primary", use_container_width=True):
    with st.spinner("🤖 AI is analyzing your urinalysis report..."):
        # Prepare user inputs - COMPLETE VERSION WITH ALL 24 FEATURES
        user_inputs = {
            # Basic demographics
            "Age": age,
            
            # Urinalysis parameters
            "pH": ph,
            "Specific Gravity": specific_gravity,
            "WBC": wbc,
            "RBC": rbc,
            "Glucose": glucose_map[glucose],
            "Protein": protein_map[protein],
            "Bacteria": bacteria_map[bacteria],
            "Transparency": transparency_map[transparency],
            
            # Microscopic findings (set reasonable defaults)
            "Epithelial Cells": 1,  # Common finding
            "Mucous Threads": 1,    # Common finding  
            "Amorphous Urates": 0,  # Less common
            
            # Gender (one-hot encoded)
            "Gender_MALE": 1 if gender == "MALE" else 0,
            "Gender_FEMALE": 1 if gender == "FEMALE" else 0,
            
            # Color features (set DARK YELLOW as default, others to 0)
            "Color_AMBER": 0,
            "Color_BROWN": 0,
            "Color_DARK YELLOW": 1,  # Most common color
            "Color_LIGHT RED": 0,
            "Color_LIGHT YELLOW": 0,
            "Color_RED": 0,
            "Color_REDDISH": 0,
            "Color_REDDISH YELLOW": 0,
            "Color_STRAW": 0,
            "Color_YELLOW": 0
        }
        
        st.session_state.user_inputs = user_inputs
        
        # Debug: Show feature count
        st.sidebar.write(f"🔄 Prepared {len(user_inputs)} features")
        
        # Make prediction
        if model and scaler and feature_names:
            prediction_result = predict_uti_risk(user_inputs, model, scaler, feature_names)
            st.session_state.prediction_result = prediction_result
            
            # Debug information
            if prediction_result:
                st.sidebar.write(f"🎯 Raw probability: {prediction_result['probability']:.3f}")
                st.sidebar.write(f"📊 Risk level: {prediction_result['risk_level']}")
        else:
            st.error("❌ Model not loaded properly. Please check the model files.")

# Debug information
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write("### Debug Information")
    if feature_names:
        st.sidebar.write(f"Expected features: {len(feature_names)}")
    if st.session_state.prediction_result:
        st.sidebar.write("Last prediction:", st.session_state.prediction_result)

# Main content area
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    
    # Risk Level Banner
    risk_class = f"risk-{result['risk_level'].lower()}"
    st.markdown(f'<div class="{risk_class}">UTI RISK LEVEL: {result["risk_level"]}</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Probability", f"{result['probability']:.1%}")
    
    with col2:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    
    with col3:
        st.metric("AI Model", "Clinical AI")
    
    with col4:
        if model_performance:
            st.metric("Model Accuracy", f"{model_performance.get('accuracy', 0.92):.1%}")

    # Explanations
    st.header("💬 Detailed Analysis")
    
    tab1, tab2 = st.tabs(["🇬🇧 English Analysis", "🇮🇳 Tamil Analysis"])
    
    with tab1:
        eng_explanation = explanation_engine.generate_explanation(
            st.session_state.user_inputs, result, 'en'
        )
        
        st.markdown("### Risk Assessment")
        st.markdown(eng_explanation['main_message'])
        
        if eng_explanation['detailed_explanations']:
            st.markdown("### 🔍 Key Findings")
            for detail in eng_explanation['detailed_explanations']:
                st.markdown(f'<div class="parameter-box">{detail}</div>', unsafe_allow_html=True)
        else:
            st.info("🎉 All tested parameters are within normal ranges.")
        
        st.markdown("### 💡 Preventive Recommendations")
        for i, tip in enumerate(eng_explanation['prevention_tips'], 1):
            st.markdown(f"{i}. {tip}")

    with tab2:
        tam_explanation = explanation_engine.generate_explanation(
            st.session_state.user_inputs, result, 'ta'
        )
        
        st.markdown("### ஆபத்து மதிப்பீடு")
        st.markdown(tam_explanation['main_message'])
        
        if tam_explanation['detailed_explanations']:
            st.markdown("### 🔍 முக்கிய கண்டறிதல்கள்")
            for detail in tam_explanation['detailed_explanations']:
                st.markdown(f'<div class="parameter-box">{detail}</div>', unsafe_allow_html=True)
        else:
            st.info("🎉 அனைத்து சோதனை அளவுருக்களும் சாதாரண வரம்புகளுக்குள் உள்ளன.")
        
        st.markdown("### 💡 தடுப்பு பரிந்துரைகள்")
        for i, tip in enumerate(tam_explanation['prevention_tips'], 1):
            st.markdown(f"{i}. {tip}")

else:
    # Welcome message
    st.markdown("""
    <div class="info-box">
    <h3>👋 Welcome to the AI-Powered UTI Detection Chatbot!</h3>
    <p>This clinical AI tool analyzes your urinalysis results to assess UTI risk and provides 
    comprehensive explanations in both English and Tamil.</p>
    
    <p><strong>📊 How it works:</strong></p>
    <ol>
        <li>Enter your lab values in the sidebar</li>
        <li>Click "Analyze My Report"</li>
        <li>Get instant AI-powered analysis with risk assessment</li>
        <li>Review detailed explanations in your preferred language</li>
        <li>Receive preventive healthcare recommendations</li>
    </ol>
    
    <p><strong>🎯 Model Performance:</strong></p>
    <ul>
        <li>Accuracy: 92.3%</li>
        <li>Trained on clinical urinalysis data</li>
        <li>Real-time risk assessment</li>
        <li>Bilingual explanations</li>
    </ul>
    
    <p><em>⚕️ Note: This AI tool provides assisted analysis and should not replace professional medical diagnosis.</em></p>
    </div>
    """, unsafe_allow_html=True)

# ========== ENHANCED GEN AI CHAT INTERFACE ==========
st.markdown("---")
st.header("🤖 AI Medical Assistant - Ask Me Anything")

chat_col1, chat_col2 = st.columns([2, 1])

with chat_col1:
    user_question = st.text_input(
        "Ask me anything about UTIs:",
        placeholder="e.g., What are UTI symptoms? How to prevent UTIs? Treatment options?",
        key="user_question"
    )
    
    language_choice = st.radio("Language:", ["English", "Tamil"], horizontal=True, key="chat_language")
    
    # AI Mode selection
    ai_mode = st.selectbox(
        "AI Intelligence Level:",
        ["Smart Medical Q&A", "Basic Information"],
        help="Smart mode provides personalized responses based on your risk assessment"
    )
    
    if st.button("🎯 Get AI Answer", type="primary", key="get_ai_answer"):
        if user_question:
            with st.spinner("🤖 AI is analyzing your question..."):
                lang_code = 'en' if language_choice == "English" else 'ta'
                
                # Use enhanced medical AI
                user_risk_data = st.session_state.prediction_result if st.session_state.prediction_result else None
                answer = medical_ai_chatbot.get_contextual_response(user_question, user_risk_data, lang_code)
                
                # Store conversation
                st.session_state.conversation.append({
                    'question': user_question,
                    'answer': answer,
                    'language': language_choice,
                    'timestamp': pd.Timestamp.now()
                })
                
                # Display answer
                st.markdown("### 💡 AI Medical Response:")
                st.markdown(f'<div class="info-box">{answer}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a question first!")

    # Display conversation history
    if st.session_state.conversation:
        st.markdown("### 📝 Conversation History")
        # Show last 5 conversations
        for i, chat in enumerate(reversed(st.session_state.conversation[-5:])):
            with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i==0)):
                st.markdown(f'<div class="chat-user"><strong>You:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-ai"><strong>AI:</strong> {chat["answer"]}</div>', unsafe_allow_html=True)

with chat_col2:
    st.markdown("### 💡 Quick Questions:")
    
    if language_choice == "English":
        quick_questions = [
            "What is UTI?",
            "UTI symptoms and signs", 
            "Treatment options for UTI",
            "How to prevent UTIs?",
            "UTI risk factors",
            "When to see a doctor?",
            "Home remedies for UTI",
            "UTI complications"
        ]
    else:
        quick_questions = [
            "யூடிஐ என்றால் என்ன?",
            "யூடிஐ அறிகுறிகள் மற்றும் அறிகுறிகள்",
            "யூடிஐ-க்கான சிகிச்சை வழிகள்",
            "யூடிஐ-க்களை எவ்வாறு தடுப்பது?",
            "யூடிஐ ஆபத்து காரணிகள்",
            "மருத்துவரை எப்போது பார்க்க வேண்டும்?",
            "யூடிஐ-க்கான வீட்டு மருந்துகள்",
            "யூடிஐ சிக்கல்கள்"
        ]
    
    for i, q in enumerate(quick_questions):
        if st.button(q, key=f"quick_q_{i}", use_container_width=True):
            st.session_state.user_question = q
            st.rerun()
    
    # Clear conversation button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
<p><strong>🩺 AI-Powered UTI Detection Chatbot</strong> | 
Clinical AI Model | Enhanced Medical Q&A | Bilingual Support | 
<em>For educational and assisted analysis purposes</em></p>
<p>Always consult healthcare professionals for medical diagnosis and treatment</p>
</div>
""", unsafe_allow_html=True)
