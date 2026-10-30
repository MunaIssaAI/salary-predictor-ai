import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# 1. إنشاء البيانات وتدريب النموذج
data = {
    'سنوات_الخبرة': [1, 2, 3, 5, 4, 6, 2, 3, 7, 5],
    'يعرف_بايثون': [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    'يعرف_تعلم_الآلة': [0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    'راتب_أعلى_من_80k': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1] 
}
df = pd.DataFrame(data)
X = df.drop('راتب_أعلى_من_80k', axis=1) 
y = df['راتب_أعلى_من_80k']
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 2. دالة التوقع
def predict_salary(experience, knows_python, knows_ml):
    py = 1 if knows_python else 0
    ml = 1 if knows_ml else 0
    
    data_input = pd.DataFrame({
        'سنوات_الخبرة': [experience],
        'يعرف_بايثون': [py],
        'يعرف_تعلم_الآلة': [ml]
    })
    
    prediction = model.predict(data_input)
    if prediction[0] == 1:
        return "💰 النتيجة: راتب أعلى من 80,000 دولار"
    else:
        return "📉 النتيجة: راتب أقل من 80,000 دولار"

# 3. إعدادات لمنع ظهور الصور العشوائية
custom_head = """
<meta property="og:title" content="AI Salary Predictor">
<meta property="og:description" content="Predict your salary based on skills.">
<meta property="og:image" content="">
<meta name="twitter:image" content="">
"""

# 4. بناء واجهة التطبيق
demo = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Slider(0, 15, value=3, label="سنوات الخبرة"),
        gr.Checkbox(label="يعرف بايثون (Python)"),
        gr.Checkbox(label="يعرف تعلم الآلة (Machine Learning)")
    ],
    outputs="text",
    title="💼 نظام التنبؤ بالرواتب",
    description="أدخل المهارات وسنقوم بتوقع الراتب!",
    head=custom_head
)

# 5. تشغيل التطبيق على منصة Render
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
