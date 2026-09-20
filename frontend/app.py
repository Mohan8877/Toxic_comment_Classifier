import requests
import streamlit as st

API_URL = st.secrets.get('API_URL', 'http://localhost:8000')

st.title('Toxic Comment Classifier - Demo')
comment = st.text_area('Enter comment', height=120)
if st.button('Analyze'):
    if not comment.strip():
        st.warning('Please enter a comment')
    else:
        resp = requests.post(f'{API_URL}/predict', json={'comment': comment})
        if resp.status_code == 200:
            data = resp.json()
            st.subheader('Result')
            st.write(f"Overall toxicity: {data['toxicity_score']*100:.1f}%")
            st.write('Categories:')
            for k, v in data['categories'].items():
                st.write(f"- {k}: {v*100:.1f}%")
            st.write('Recommendation:')
            st.info(data['moderation_action'])
        else:
            st.error('Prediction failed')
