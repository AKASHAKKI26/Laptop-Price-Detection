import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error


st.title("🛒 Amazon Price Prediction App")

def load_data():
    data = pd.read_csv("Price_detection")
    data["Price"] = (data["Price_USD"] * 90.94).round().astype(int)
    data.drop(columns=["Price_USD"], inplace=True)
    return data

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

data["RAM_GB"] = data["RAM_GB"].astype(int)
data["Storage_GB"] = data["Storage_GB"].astype(int)

data = pd.get_dummies(data, drop_first=True)

x = data.drop("Price", axis=1)
y = data["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
st.write("R-Squared Score: ",round(r2,3))

st.subheader("Enter Product Details")

b=st.selectbox("Select Brand:",["Acer","Apple","Asus","Dell","HP","Lenovo","MS"])
p=st.selectbox("Select Processor:",['AMD Ryzen 3', 'Intel i7', 'AMD Ryzen 7', 'Intel i5', 'Intel i3','AMD Ryzen 5'])
ram=st.selectbox("Select RAM",[ 4, 32,  8, 16])
s=st.selectbox("Select Storage",[ 512,  128,  256, 1024])
o=st.selectbox("Select OS",['macOS', 'Windows 10', 'Windows 11'])
g=st.selectbox("Select GPU",['AMD Radeon', 'NVIDIA GTX 1650', 'Integrated', 'NVIDIA RTX 3050'])
r=st.number_input("Enter the Rating",min_value=0.0,max_value=5.0)

input_data = pd.DataFrame({
    "Brand": [b],
    "Processor": [p],
    "RAM_GB": [ram],
    "Storage_GB": [s],
    "OS": [o],
    "GPU": [g],
    "Rating": [r]
})
if st.button("Predict Price"):
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=x.columns, fill_value=0)
    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹ {round(prediction[0], 2)}")
