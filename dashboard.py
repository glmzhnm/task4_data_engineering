import os
import re
import yaml
import pandas as pd
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime


def fix_price(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    cleann = re.sub(r'[^\d,\.]', '', val_str).replace(',', '.')
    if not cleann:
        return 0.0
    res = float(cleann)
    if '€' in val_str:
        res = res * 1.2
    return round(res, 2)


def get_users(df):
    df = df.fillna('').astype(str)
    g = nx.Graph()
    g.add_nodes_from(df['id'].tolist())

    combs = [
        ['name', 'address', 'phone'],
        ['name', 'address', 'email'],
        ['name', 'phone', 'email'],
        ['address', 'phone', 'email']
    ]

    for i in combs:
        groups = defaultdict(list)
        for j, row in df.iterrows():
            key = tuple([row[x].strip().lower() for x in i])
            if any(key):
                groups[key].append(row['id'])

        for key, ids in groups.items():
            if len(ids) > 1:
                for j in range(len(ids) - 1):
                    g.add_edge(ids[j], ids[j + 1])

    components = list(nx.connected_components(g))
    mapping = {}
    for idx, comp in enumerate(components):
        for uid in comp:
            mapping[uid] = str(idx)

    return mapping, components


def process_folder(path):
    users = pd.read_csv(os.path.join(path, 'users.csv'))
    mapping, components = get_users(users)

    orders = pd.read_parquet(os.path.join(path, 'orders.parquet'))

    orders['timestamp'] = pd.to_datetime(orders['timestamp'].astype(str), errors='coerce', utc=True)
    orders['date'] = orders['timestamp'].dt.date
    current_year = datetime.datetime.now().year

    def fix_year(d):
        if pd.isna(d): return d
        if d.year > 2025:
            return d.replace(year=2024)
        return d

    orders['date'] = orders['date'].apply(fix_year)
    orders['unit_price'] = orders['unit_price'].apply(fix_price)
    orders['quantity'] = pd.to_numeric(orders['quantity'], errors='coerce').fillna(0)
    orders['paid_price'] = orders['quantity'] * orders['unit_price']

    ucol = 'user_id' if 'user_id' in orders.columns else 'id'
    orders[ucol] = orders[ucol].astype(str)
    orders['real_user'] = orders[ucol].map(mapping)

    with open(os.path.join(path, 'books.yaml'), encoding='utf-8') as f:
        b_data = yaml.safe_load(f)
    books = pd.DataFrame(b_data)

    books.columns = [c.replace(':', '') for c in books.columns]
    books['id'] = books['id'].astype(str)

    def parse_authors(a):
        if pd.isna(a):
            return tuple()
        if isinstance(a, list):
            lst = [str(x).strip() for x in a]
        else:
            lst = [x.strip() for x in str(a).split(',')]
        return tuple(sorted(lst))

    books['author_set'] = books['author'].apply(parse_authors)

    return users, orders, books, components


st.set_page_config(layout="wide")
st.title("Orders Analytics")

tabs = st.tabs(["DATA 1", "DATA 2", "DATA 3"])
folders = ["DATA1", "DATA2", "DATA3"]

for i, ds in enumerate(folders):
    with tabs[i]:
        path = f"data/{ds}"

        if not os.path.exists(path):
            st.error(f"Folder not found: {path}")
            continue

        users, orders, books, comps = process_folder(path)

        rev_by_day = orders.groupby('date')['paid_price'].sum().reset_index()
        top5 = rev_by_day.sort_values('paid_price', ascending=False).head(5)
        top5['date'] = pd.to_datetime(top5['date']).dt.strftime('%Y-%m-%d')

        uniq_users = len(comps)
        uniq_authors = books['author_set'].nunique()

        b_col = 'book_id' if 'book_id' in orders.columns else 'id'
        orders[b_col] = orders[b_col].astype(str)

        m = orders.merge(books, left_on=b_col, right_on='id', how='left')
        pop_author = m.groupby('author_set')['quantity'].sum().idxmax()

        if pop_author:
            pop_author_str = ", ".join(pop_author)
        else:
            pop_author_str = "Unknown"

        spend = orders.groupby('real_user')['paid_price'].sum()
        best_id = spend.idxmax()
        aliases = list(comps[int(best_id)])

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Metrics")
            st.write(f"- **Unique users:** {uniq_users}")
            st.write(f"- **Unique author sets:** {uniq_authors}")
            st.write(f"- **Most popular author:** {pop_author_str}")
            st.write(f"- **Best buyer (aliases):** `{aliases}`")

            st.write("### Top 5 days by revenue")
            st.dataframe(top5, hide_index=True)

        with col2:
            st.write("### Chart")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(pd.to_datetime(rev_by_day['date']), rev_by_day['paid_price'], color='tab:blue')
            ax.set_ylabel("Revenue ($)")
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

        st.divider()