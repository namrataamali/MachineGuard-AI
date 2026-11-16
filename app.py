from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='MachineGuard AI | Industrial Intelligence', page_icon='⚙️', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif;color:#e9edf8}
.stApp{background:radial-gradient(ellipse at 80% 0%,#1d2844 0%,#0b101c 43%,#080b13 100%)}
[data-testid="stSidebar"]{background:#0d1321;border-right:1px solid #253046}
[data-testid="stHeader"]{background:transparent}
h1,h2,h3{font-family:'Outfit',sans-serif;letter-spacing:-.03em}
.hero{padding:42px 42px 38px;border-radius:22px;background:linear-gradient(105deg,rgba(8,15,28,.98) 0%,rgba(18,31,52,.9) 57%,rgba(29,69,95,.6) 100%),radial-gradient(circle at 95% 20%,#30c4c5 0%,transparent 50%);border:1px solid #2b3c55;margin-bottom:20px;position:relative;overflow:hidden}
.hero:after{content:'◉';position:absolute;right:7%;top:-90px;font-size:330px;color:rgba(68,224,219,.08);pointer-events:none}
.eyebrow{color:#6ce7dc;font-size:12px;font-weight:800;letter-spacing:3px;margin-bottom:15px}
.hero h1{font-size:clamp(35px,5vw,64px);line-height:1.05;margin:0 0 16px;color:#fff;font-weight:800}
.hero p{max-width:630px;color:#b6c5d7;font-size:16px;line-height:1.7}
.tag{display:inline-block;padding:7px 12px;border:1px solid #386e77;background:#16343f;color:#7ff3e8;border-radius:30px;font-size:11px;font-weight:700;margin:8px 6px 0 0}
.section-label{color:#7d91ac;font-size:12px;font-weight:800;letter-spacing:2px;margin-top:20px}
.metric-card{padding:21px 24px;border-radius:16px;background:linear-gradient(145deg,#172235,#111a2b);border:1px solid #2a3850;min-height:132px}
.metric-card .label{color:#98a8bd;font-size:12px;letter-spacing:.7px;text-transform:uppercase}
.metric-card .value{font-size:35px;font-family:Outfit;font-weight:800;color:white;margin:8px 0}
.metric-card .hint{color:#5ee2d4;font-size:12px}
.stButton>button[kind="primary"]{background:#55d9cb;color:#08151d;border:0;font-weight:800;border-radius:9px}
.stButton>button{border-radius:9px}
[data-testid="stFileUploader"],div[data-testid="stForm"]{border-color:#2a3850}
.status{padding:20px;border-radius:15px;border:1px solid #33435b;background:#121e30;margin:15px 0}
.status strong{font-size:21px}
small{color:#91a4ba}
</style>''', unsafe_allow_html=True)

FEATURES = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_H', 'Type_L', 'Type_M']

@st.cache_resource(show_spinner='Preparing MachineGuard prediction engine...')
def load_model():
    model_path = BASE / 'machineguard_model.pkl'
    features_path = BASE / 'feature_names.pkl'
    if not model_path.exists() or not features_path.exists():
        from train_model import train
        train()
    return joblib.load(model_path), joblib.load(features_path)

try:
    model, feature_names = load_model()
except Exception as exc:
    st.error('Could not initialize the model. Internet is required on the first launch, or place your trained machineguard_model.pkl and feature_names.pkl beside app.py.')
    st.exception(exc)
    st.stop()

def assess(df):
    frame = df.reindex(columns=feature_names, fill_value=0)
    probs = model.predict_proba(frame)[:, 1]
    results = df.copy()
    results['Failure Probability (%)'] = (probs * 100).round(1)
    results['Risk Level'] = pd.cut(probs, bins=[-0.001, .25, .60, 1.001], labels=['Low Risk', 'Medium Risk', 'High Risk'], right=False).astype(str)
    def advice(row):
        notes=[]
        if row['Tool wear [min]'] > 200: notes.append('Inspect tool wear and consider replacement')
        if row['Torque [Nm]'] > 55: notes.append('Check mechanical load and torque')
        if row['Rotational speed [rpm]'] < 1300: notes.append('Inspect motor and drive system')
        if row['Process temperature [K]'] > 315: notes.append('Inspect cooling and temperature control')
        if row['Risk Level']=='High Risk': notes.append('Prioritize technician inspection')
        if not notes: notes.append('Continue routine monitoring')
        return ' • '.join(notes)
    results['Recommendations'] = results.apply(advice, axis=1)
    return results

with st.sidebar:
    st.markdown('## ⚙️ MACHINEGUARD')
    st.caption('INDUSTRIAL INTELLIGENCE PLATFORM')
    st.divider()
    page = st.radio('NAVIGATION', ['Overview', 'Machine Assessment', 'Batch Intelligence', 'Model & About'], label_visibility='visible')
    st.divider()
    st.success('● Prediction engine online')
    st.caption('Model: Random Forest • 100 trees')
    st.caption('Data: UCI AI4I 2020 (synthetic)')
    st.caption('Risk bands and recommendations are demonstration rules, not certified industrial safety guidance.')

if page=='Overview':
    st.markdown('''<div class="hero"><div class="eyebrow">NEXT-GEN PREDICTIVE MAINTENANCE</div><h1>Stay ahead of<br>machine failure.</h1><p>From raw sensor readings to actionable maintenance intelligence. Predict failure risk, detect warning signs, and prioritize inspections — in one command center.</p><span class="tag">● LIVE MODEL READY</span><span class="tag">RISK INTELLIGENCE</span><span class="tag">MAINTENANCE GUIDANCE</span></div>''',unsafe_allow_html=True)
    cols=st.columns(3)
    for col,label,value,hint in zip(cols,['MODEL TYPE','TRAINING RECORDS','FAILURE RATE IN DATA'],['Random Forest','10,000','3.39%'],['Binary classification','UCI AI4I 2020','339 labeled failures']):
        with col: st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>',unsafe_allow_html=True)
    st.markdown('### Explore the platform')
    a,b=st.columns(2)
    with a:
        st.info('**⚡ Machine Assessment**  \nEnter sensor measurements and get a failure score, risk band, and suggested checks.')
    with b:
        st.info('**📂 Batch Intelligence**  \nUpload a CSV, analyze a fleet of machines, and download a risk-enriched report.')
    st.caption('The interface is inspired by cinematic dark layouts; it does not reproduce third-party branding or content.')

elif page=='Machine Assessment':
    st.markdown('''<div class="hero"><div class="eyebrow">SENSOR-BASED DIAGNOSTICS</div><h1>Machine Assessment</h1><p>Enter current operating conditions to generate a machine failure risk estimate and recommended inspection actions.</p></div>''',unsafe_allow_html=True)
    with st.form('machine'):
        c1,c2,c3=st.columns(3)
        with c1:
            machine_type=st.selectbox('Machine type',['M','L','H'])
            air=st.number_input('Air temperature (K)',value=300.0,step=.1)
        with c2:
            process=st.number_input('Process temperature (K)',value=310.0,step=.1)
            speed=st.number_input('Rotational speed (RPM)',min_value=0,value=1500)
        with c3:
            torque=st.number_input('Torque (Nm)',min_value=0.0,value=40.0,step=.1)
            wear=st.number_input('Tool wear (minutes)',min_value=0,value=100)
        submitted=st.form_submit_button('⚡ ANALYZE MACHINE',type='primary',use_container_width=True)
    if submitted:
        record={'Air temperature [K]':air,'Process temperature [K]':process,'Rotational speed [rpm]':speed,'Torque [Nm]':torque,'Tool wear [min]':wear,'Type_H':int(machine_type=='H'),'Type_L':int(machine_type=='L'),'Type_M':int(machine_type=='M')}
        result=assess(pd.DataFrame([record])).iloc[0]
        risk=result['Risk Level']; prob=result['Failure Probability (%)']
        color={'Low Risk':'#65e1ca','Medium Risk':'#f5c66b','High Risk':'#ff7b8e'}[risk]
        st.markdown(f'<div class="status"><small>MODEL-ESTIMATED FAILURE PROBABILITY</small><h1 style="color:{color};margin:4px 0">{prob:.1f}%</h1><strong style="color:{color}">{risk.upper()}</strong></div>',unsafe_allow_html=True)
        st.progress(min(float(prob)/100,1.0))
        st.markdown('#### 🔧 Recommended actions')
        for item in result['Recommendations'].split(' • '): st.write('• '+item)
        st.caption('This score is not a calibrated real-world failure probability. Recommendations are simple illustrative rules and must not replace qualified maintenance inspections.')

elif page=='Batch Intelligence':
    st.markdown('''<div class="hero"><div class="eyebrow">FLEET-WIDE RISK ANALYTICS</div><h1>Batch Intelligence</h1><p>Upload machine readings, compare failure risk across the fleet, and export a maintenance-ready assessment.</p></div>''',unsafe_allow_html=True)
    sample=pd.DataFrame([
        {'Type':'M','Air temperature [K]':300.0,'Process temperature [K]':310.0,'Rotational speed [rpm]':1500,'Torque [Nm]':40.0,'Tool wear [min]':100},
        {'Type':'L','Air temperature [K]':305.0,'Process temperature [K]':315.0,'Rotational speed [rpm]':1250,'Torque [Nm]':60.0,'Tool wear [min]':220},
        {'Type':'H','Air temperature [K]':298.0,'Process temperature [K]':308.0,'Rotational speed [rpm]':1650,'Torque [Nm]':35.0,'Tool wear [min]':40}
    ])
    st.download_button('⬇ Download sample CSV',sample.to_csv(index=False).encode(),file_name='machineguard_sample.csv',mime='text/csv')
    upload=st.file_uploader('Upload a machine CSV',type=['csv'])
    if upload is not None:
        try:
            incoming=pd.read_csv(upload)
            required=['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
            missing=[c for c in required if c not in incoming.columns]
            if missing: raise ValueError('Missing columns: '+', '.join(missing))
            if 'Type' in incoming.columns:
                machine_types=incoming['Type'].astype(str).str.upper().str.strip()
                if not machine_types.isin(['L','M','H']).all(): raise ValueError('Type must be L, M or H.')
                for t in ['H','L','M']: incoming['Type_'+t]=(machine_types==t).astype(int)
            elif not all('Type_'+t in incoming.columns for t in ['H','L','M']):
                raise ValueError('Provide Type (L/M/H) or all three Type_H, Type_L, Type_M columns.')
            for c in required: incoming[c]=pd.to_numeric(incoming[c],errors='raise')
            if incoming[required].isna().any().any(): raise ValueError('Sensor measurements must not contain missing values.')
            if (incoming[['Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']]<0).any().any(): raise ValueError('Speed, torque and wear cannot be negative.')
            results=assess(incoming)
            counts=results['Risk Level'].value_counts()
            a,b,c=st.columns(3)
            a.metric('Machines analyzed',len(results))
            b.metric('High risk',int(counts.get('High Risk',0)))
            c.metric('Average failure score',f"{results['Failure Probability (%)'].mean():.1f}%")
            graph=px.bar(x=['Low Risk','Medium Risk','High Risk'],y=[int(counts.get(k,0)) for k in ['Low Risk','Medium Risk','High Risk']],color=['Low Risk','Medium Risk','High Risk'],color_discrete_map={'Low Risk':'#58d9c8','Medium Risk':'#efbb62','High Risk':'#ff738a'},labels={'x':'Risk level','y':'Machine count'},template='plotly_dark')
            graph.update_layout(showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(graph,use_container_width=True)
            st.dataframe(results,use_container_width=True)
            st.download_button('⬇ Download assessment report',results.to_csv(index=False).encode(),file_name='machineguard_assessment.csv',mime='text/csv',type='primary')
        except Exception as exc:
            st.error(f'Unable to process file: {exc}')

else:
    st.markdown('### Model & Project Information')
    st.write('**Algorithm:** RandomForestClassifier (100 trees, class_weight="balanced", random_state=42)')
    st.write('**Dataset:** UCI AI4I 2020 Predictive Maintenance (10,000 synthetic records)')
    st.write('**Inputs:** machine type, air/process temperature, rotational speed, torque and tool wear')
    st.write('**Failure classification threshold:** 25% (experimental)')
    st.write('**Risk levels:** Low <25%; Medium 25–<60%; High ≥60%')
    st.write('**Recommendation engine:** Explicit, illustrative rules based on sensor values')
    st.warning('The initial 25% threshold was chosen after examining test-set performance. For unbiased evaluation, tune thresholds on a separate validation set and report final performance on untouched test data. Model scores are not calibrated operational probabilities.')
    st.markdown('[UCI dataset source](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)')
