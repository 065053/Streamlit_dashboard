
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import io
import gdown

# Load data from Google Drive
file_id = "1mEe7-GM94f67ybcoz9F9YOlQ_yW7iO-z"
url = f"https://drive.google.com/uc?id=1mEe7-GM94f67ybcoz9F9YOlQ_yW7iO-z"

output = "new_retail_data.csv"
gdown.download(url, output, quiet=False, fuzzy=True)

df = pd.read_csv(output)

# Define columns
categorical_cols = [
    'City', 'State', 'Country', 'Gender', 'Income', 'Customer_Segment', 'Month',
    'Product_Category', 'Product_Brand', 'Product_Type', 'Feedback',
    'Shipping_Method', 'Payment_Method', 'Order_Status'
]
non_categorical_cols = [
    'Transaction_ID', 'Customer_ID', 'Name', 'Email', 'Phone', 'Address', 'Zipcode',
    'Age', 'Date', 'Year', 'Time', 'Total_Purchases', 'Amount', 'Total_Amount',
    'Ratings', 'products'
]



# Sidebar filters
st.sidebar.title('Dashboard Filters')

# Year filter
year_options = df['Year'].dropna().unique()
selected_years = st.sidebar.multiselect('Select Year(s)', sorted(year_options))

# Amount filter
amount_min, amount_max = int(df['Amount'].min()), int(df['Amount'].max())
amount_range = st.sidebar.slider('Amount Range', min_value=amount_min, max_value=amount_max, value=(amount_min, amount_max))

# Age filter
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider('Age Range', min_value=age_min, max_value=age_max, value=(age_min, age_max))

# Products filter
product_options = df['products'].dropna().unique()
selected_products = st.sidebar.multiselect('Select Product(s)', sorted(product_options))

# State filter
state_options = df['State'].dropna().unique()
selected_states = st.sidebar.multiselect('Select State(s)', sorted(state_options))

# Product_Brand filter
brand_options = df['Product_Brand'].dropna().unique()
selected_brands = st.sidebar.multiselect('Select Product Brand(s)', sorted(brand_options))

# Product_Type filter
type_options = df['Product_Type'].dropna().unique()
selected_types = st.sidebar.multiselect('Select Product Type(s)', sorted(type_options))

# Shipping_Method filter
shipping_options = df['Shipping_Method'].dropna().unique()
selected_shipping = st.sidebar.multiselect('Select Shipping Method(s)', sorted(shipping_options))

# Payment_Method filter
payment_options = df['Payment_Method'].dropna().unique()
selected_payment = st.sidebar.multiselect('Select Payment Method(s)', sorted(payment_options))

# Order_Status filter
order_status_options = df['Order_Status'].dropna().unique()
selected_order_status = st.sidebar.multiselect('Select Order Status(es)', sorted(order_status_options))

# Feedback filter
feedback_options = df['Feedback'].dropna().unique()
selected_feedback = st.sidebar.multiselect('Select Feedback(s)', sorted(feedback_options))

# Month filter
month_options = df['Month'].dropna().unique()
selected_months = st.sidebar.multiselect('Select Month(s)', sorted(month_options))

# Country filter
country_options = df['Country'].dropna().unique()
selected_countries = st.sidebar.multiselect('Select Country(ies)', sorted(country_options))

# Gender filter
gender_options = df['Gender'].dropna().unique()
selected_genders = st.sidebar.multiselect('Select Gender(s)', sorted(gender_options))

# Income filter
income_options = df['Income'].dropna().unique()
selected_incomes = st.sidebar.multiselect('Select Income(s)', sorted(income_options))

# Product_Category filter
product_cat_options = df['Product_Category'].dropna().unique()
selected_product_cats = st.sidebar.multiselect('Select Product Category(ies)', sorted(product_cat_options))

# Ratings filter
rating_min, rating_max = int(df['Ratings'].min()), int(df['Ratings'].max())
rating_range = st.sidebar.slider('Ratings Range', min_value=rating_min, max_value=rating_max, value=(rating_min, rating_max))


# Filter the DataFrame based on selections
filtered_df = df.copy()
if selected_years:
    filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
filtered_df = filtered_df[(filtered_df['Amount'] >= amount_range[0]) & (filtered_df['Amount'] <= amount_range[1])]
filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
if selected_products:
    filtered_df = filtered_df[filtered_df['products'].isin(selected_products)]
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
if selected_brands:
    filtered_df = filtered_df[filtered_df['Product_Brand'].isin(selected_brands)]
if selected_types:
    filtered_df = filtered_df[filtered_df['Product_Type'].isin(selected_types)]
if selected_shipping:
    filtered_df = filtered_df[filtered_df['Shipping_Method'].isin(selected_shipping)]
if selected_payment:
    filtered_df = filtered_df[filtered_df['Payment_Method'].isin(selected_payment)]
if selected_order_status:
    filtered_df = filtered_df[filtered_df['Order_Status'].isin(selected_order_status)]
if selected_feedback:
    filtered_df = filtered_df[filtered_df['Feedback'].isin(selected_feedback)]
filtered_df = filtered_df[(filtered_df['Ratings'] >= rating_range[0]) & (filtered_df['Ratings'] <= rating_range[1])]
if selected_months:
    filtered_df = filtered_df[filtered_df['Month'].isin(selected_months)]
if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
if selected_genders:
    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
if selected_incomes:
    filtered_df = filtered_df[filtered_df['Income'].isin(selected_incomes)]
if selected_product_cats:
    filtered_df = filtered_df[filtered_df['Product_Category'].isin(selected_product_cats)]

st.title('New Retail Data Dashboard')




import matplotlib.pyplot as plt
import seaborn as sns

# Split categorical columns for different chart types
half_index = len(categorical_cols) // 2
pie_chart_cols = categorical_cols[:half_index]
bar_chart_cols = categorical_cols[half_index:]

# Remove rows with nulls for plotting, but use filtered_df so pie charts react to filters
filtered_rows_without_nulls = filtered_df.dropna()


# Pie Charts for the first half of categorical columns (3 per row)
st.header('Pie Charts for Categorical Variables')
pie_cols = st.columns(3)
for i, col in enumerate(pie_chart_cols):
    with pie_cols[i % 3]:
        st.subheader(f"{col}")
        pie_counts = filtered_rows_without_nulls[col].value_counts()
        pie_top_n = pie_counts.head(10)  # Display top 10 for clarity in pie charts
        plt.figure(figsize=(8, 8))
        plt.pie(pie_top_n, labels=pie_top_n.index, autopct='%1.1f%%', startangle=140, wedgeprops={'width':0.6})
        plt.title(f'Distribution of {col}')
        plt.axis('equal')
        st.pyplot(plt.gcf())
        plt.clf()
    # Start a new row every 3 charts
    if (i % 3 == 2) and (i != len(pie_chart_cols) - 1):
        pie_cols = st.columns(3)


st.header('Selected Box Plots')
# Restore selected_box_plots definition
selected_box_plots = [
    ('Age', 'Income'),
    ('Total_Amount', 'Customer_Segment'),
    ('Ratings', 'Product_Category'),
    ('Total_Purchases', 'Country')
]
box_cols = st.columns(3)
for i, (num_col, cat_col) in enumerate(selected_box_plots):
    with box_cols[i % 3]:
        st.subheader(f"Box Plot for {num_col} by {cat_col}")
        plt.figure(figsize=(8, 8))
        sns.boxplot(data=filtered_df, x=cat_col, y=num_col)
        plt.title(f'{num_col} by {cat_col}')
        plt.xticks(rotation=90)
        st.pyplot(plt.gcf())
        plt.clf()
    if (i % 3 == 2) and (i != len(selected_box_plots) - 1):
        box_cols = st.columns(3)
st.header('Heatmap of Numerical Feature Correlations')
heatmap_cols = st.columns(3)


import altair as alt


melt_columns = [col for col in ['Total_Purchases', 'Total_Amount', 'Ratings', 'Age', 'Feedback'] if col in filtered_df.columns]
product_category_df = filtered_df.groupby('Product_Category')[melt_columns].mean().round(1)


# Melt the dataframe to long format for Altair
melted_product_category_df = product_category_df.reset_index().melt(
    'Product_Category', var_name='Metric', value_name='Value'
)








# Unified Heatmaps Section
heatmap_cols = st.columns(3)
heatmap_idx = 0

# 1. Correlation heatmap of numerical features
with heatmap_cols[heatmap_idx % 3]:
    st.subheader('Correlation Heatmap of Numerical Features')
    numerical_cols = filtered_df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 1:
        corr_matrix = filtered_df[numerical_cols].corr()
        plt.figure(figsize=(8, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Numerical Features')
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.write('Not enough numerical columns to display a correlation heatmap.')
heatmap_idx += 1

# 2. Mean values by Top 10 Product Brands
with heatmap_cols[heatmap_idx % 3]:
    st.subheader('Mean Values by Top 10 Product Brands')
    top_product_brands = filtered_df.groupby('Product_Brand')['Total_Amount'].sum().nlargest(10).index
    product_brand_heatmap_data = filtered_df[filtered_df['Product_Brand'].isin(top_product_brands)].groupby('Product_Brand')[['Total_Purchases', 'Total_Amount', 'Ratings', 'Age', 'Feedback']].mean().round(1)
    plt.figure(figsize=(8, 8))
    sns.heatmap(product_brand_heatmap_data.transpose(), annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Mean Values by Top 10 Product Brands')
    st.pyplot(plt.gcf())
    plt.clf()
heatmap_idx += 1

# 3. Mean values by Top 10 Product Categories
with heatmap_cols[heatmap_idx % 3]:
    st.subheader('Mean Values by Top 10 Product Categories')
    top_product_cats = filtered_df.groupby('Product_Category')['Total_Amount'].sum().nlargest(10).index
    product_cat_heatmap_data = filtered_df[filtered_df['Product_Category'].isin(top_product_cats)].groupby('Product_Category')[['Total_Purchases', 'Total_Amount', 'Ratings', 'Age', 'Feedback']].mean().round(1)
    plt.figure(figsize=(8, 8))
    sns.heatmap(product_cat_heatmap_data.transpose(), annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Mean Values by Top 10 Product Categories')
    st.pyplot(plt.gcf())
    plt.clf()
heatmap_idx += 1
    
if heatmap_idx % 3 == 0:
    heatmap_cols = st.columns(3)

# 4. Mean values by Top 10 Product Types
with heatmap_cols[heatmap_idx % 3]:
    st.subheader('Mean Values by Top 10 Product Types')
    top_product_types = filtered_df.groupby('Product_Type')['Total_Amount'].sum().nlargest(10).index
    product_type_heatmap_data = filtered_df[filtered_df['Product_Type'].isin(top_product_types)].groupby('Product_Type')[['Total_Purchases', 'Total_Amount', 'Ratings', 'Age', 'Feedback']].mean().round(1)
    plt.figure(figsize=(8, 8))
    sns.heatmap(product_type_heatmap_data.transpose(), annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Mean Values by Top 10 Product Types')
    st.pyplot(plt.gcf())
    plt.clf()
heatmap_idx += 1

if heatmap_idx % 3 == 0:
    heatmap_cols = st.columns(3)

# 5. Mean values by Customer Segment
with heatmap_cols[heatmap_idx % 3]:
    st.subheader('Mean Values by Customer Segment')
    segment_columns = [col for col in ['Total_Purchases', 'Total_Amount', 'Ratings', 'Age', 'Feedback'] if col in filtered_df.columns]
    customer_segment_heatmap_data = filtered_df.groupby('Customer_Segment')[segment_columns].mean().round(1)
    plt.figure(figsize=(8, 8))
    sns.heatmap(customer_segment_heatmap_data.transpose(), annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Mean Values by Customer Segment')
    st.pyplot(plt.gcf())
    plt.clf()
heatmap_idx += 1

if heatmap_idx % 3 == 0:
    heatmap_cols = st.columns(3)

# 6. Mean values by Income
with heatmap_cols[heatmap_idx % 3]:
    st.subheader('Mean Values by Income')
    income_columns = [col for col in ['Total_Purchases', 'Total_Amount', 'Ratings', 'Age', 'Feedback'] if col in filtered_df.columns]
    income_heatmap_data = filtered_df.groupby('Income')[income_columns].mean().round(1)
    plt.figure(figsize=(8, 8))
    sns.heatmap(income_heatmap_data.transpose(), annot=True, cmap='coolwarm', fmt='.1f')
    plt.title('Mean Values by Income')
    st.pyplot(plt.gcf())
    plt.clf()
heatmap_idx += 1

if heatmap_idx % 3 == 0:
    heatmap_cols = st.columns(3)

import altair as alt

# Group by Product_Brand and calculate mean values using filtered_df
brand_columns = [col for col in ['Total_Purchases', 'Total_Amount', 'Ratings', 'Age', 'Feedback'] if col in filtered_df.columns]
product_brand_df = filtered_df.groupby('Product_Brand')[brand_columns].mean().round(1)



import altair as alt

# Only use 'Total_Purchases' and 'Ratings' columns if they exist in filtered_df
type_columns = [col for col in ['Total_Purchases', 'Ratings'] if col in filtered_df.columns]
product_type_df = filtered_df.groupby('Product_Type')[type_columns].mean().reset_index()

# Melt the dataframe to long format for Altair
melted_product_type_df = product_type_df.melt(
    'Product_Type', var_name='Metric', value_name='Value'
)

# Create a lollipop (mentos) plot for Total_Purchases and Ratings by Product Type
base = alt.Chart(melted_product_type_df).encode(
    x=alt.X('Product_Type:N', title='Product Type', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('Value:Q', title='Average Value'),
    color=alt.Color('Metric:N', title='Metric'),
    tooltip=['Product_Type', 'Metric', 'Value']
)

# Line (stick)
lines = base.mark_rule().encode(
    y=alt.Y('Value:Q'),
    y2=alt.value(0)
)
# Dot (mentos)
dots = base.mark_circle(size=120)

lollipop = (lines + dots).properties(
    width=40 * max(1, len(product_type_df['Product_Type'])),
    height=350,
    title='Average Total Purchases and Ratings by Product Type (Lollipop Plot)'
)

st.header('Average Total Purchases and Ratings by Product Type (Lollipop Plot)')
st.altair_chart(lollipop, use_container_width=True)





import matplotlib.pyplot as plt
import seaborn as sns

# Only use columns that exist in filtered_df
violin_columns = [col for col in ['Ratings', 'feedback_number', 'Total_Purchases', 'Age'] if col in filtered_df.columns]

# Create a violin plot for each metric by Income (3 per row)
violin_group_cols = st.columns(3)
for i, metric in enumerate(violin_columns):
    with violin_group_cols[i % 3]:
        plt.figure(figsize=(8, 6))
        sns.violinplot(data=filtered_df, x='Income', y=metric, inner='quartile')
        plt.title(f'{metric} Distribution by Income')
        plt.xlabel('Income')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.clf()
    if (i % 3 == 2) and (i != len(violin_columns) - 1):
        violin_group_cols = st.columns(3)

        # === GROUPED BAR CHARTS (ALTAIR) AT THE END ===

# 1. Total Purchases by Product Category (Altair)
melted_product_category_df = product_category_df.reset_index().melt(
    'Product_Category', var_name='Metric', value_name='Value'
)
chart = alt.Chart(melted_product_category_df).mark_bar(size=35).encode(
    x=alt.X('Product_Category', axis=None),
    y='Value',
    color='Metric',
    column=alt.Column('Product_Category', header=alt.Header(titleOrient="bottom", labelOrient="bottom"), spacing=1),
    tooltip=['Product_Category', 'Metric', 'Value']
).properties(
    title='Total Purchases by Product Category'
).configure_view(
    step=5
).interactive()
st.header('Total Purchases by Product Category (Altair)')
st.altair_chart(chart, use_container_width=True)

# 2. Mean Numerical Values by Product Brand (Altair)
product_brand_df = filtered_df.groupby('Product_Brand')[['Total_Purchases', 'Total_Amount', 'Ratings', 'Age']].mean().round(1)
product_brand_df_for_chart = product_brand_df.reset_index()
melted_product_brand_df = product_brand_df_for_chart.melt(
    'Product_Brand', var_name='Metric', value_name='Value'
)
melted_product_brand_df = melted_product_brand_df[melted_product_brand_df['Metric'] != 'Total_Amount']
chart = alt.Chart(melted_product_brand_df).mark_bar(size=35).encode(
    x=alt.X('Metric', axis=None),
    y='Value',
    color='Metric',
    column=alt.Column('Product_Brand', header=alt.Header(titleOrient="bottom", labelOrient="bottom"), spacing=1),
    tooltip=['Product_Brand', 'Metric', 'Value']
).properties(
    title='Mean Numerical Values by Product Brand (Excluding Total Amount)'
).configure_view(
    step=5
).interactive()
st.header('Mean Numerical Values by Product Brand (Altair)')
st.altair_chart(chart, use_container_width=True)



# 4. Average Ratings, Feedback Number, Total Purchases, and Age by Customer Segment (Altair)
segment_columns = [col for col in ['Total_Amount', 'Ratings', 'feedback_number', 'Total_Purchases', 'Age'] if col in filtered_df.columns]
customer_segment_stats = filtered_df.groupby('Customer_Segment')[segment_columns].mean().reset_index()
melted_customer_segment_stats = customer_segment_stats.melt(
    'Customer_Segment', var_name='Metric', value_name='Value'
)
melted_customer_segment_stats = melted_customer_segment_stats[melted_customer_segment_stats['Metric'] != 'Total_Amount']
chart = alt.Chart(melted_customer_segment_stats).mark_bar(size=35).encode(
    x=alt.X('Metric', axis=None),
    y='Value',
    color='Metric',
    column=alt.Column('Customer_Segment', header=alt.Header(titleOrient="bottom", labelOrient="bottom"), spacing=1),
    tooltip=['Customer_Segment', 'Metric', 'Value']
).properties(
    title='Average Ratings, Feedback Number, Total Purchases, and Age by Customer Segment'
).configure_view(
    step=5
).interactive()
st.header('Average Ratings, Feedback Number, Total Purchases, and Age by Customer Segment (Altair)')
st.altair_chart(chart, use_container_width=True)

# 5. Average Ratings, Feedback Number, Total Purchases, and Age by Income (Altair)
income_columns = [col for col in ['Ratings', 'feedback_number', 'Total_Purchases', 'Age'] if col in filtered_df.columns]
income_stats = filtered_df.groupby('Income')[income_columns].mean().reset_index()
melted_income_stats = income_stats.melt(
    'Income', var_name='Metric', value_name='Value'
)
chart = alt.Chart(melted_income_stats).mark_bar(size=35).encode(
    x=alt.X('Metric', axis=None),
    y='Value',
    color='Metric',
    column=alt.Column('Income', header=alt.Header(titleOrient="bottom", labelOrient="bottom"), spacing=1),
    tooltip=['Income', 'Metric', 'Value']
).properties(
    title='Average Ratings, Feedback Number, Total Purchases, and Age by Income'
).configure_view(
    step=5
).interactive()
st.header('Average Ratings, Feedback Number, Total Purchases, and Age by Income (Altair)')
st.altair_chart(chart, use_container_width=True)