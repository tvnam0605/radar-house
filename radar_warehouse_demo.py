import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import io

# -------------------------
# Helper / Utils
# -------------------------
@st.cache_data
def generate_mock_data(n=500):
    np.random.seed(42)
    warehouses = ["WH_A", "WH_B", "WH_C", "WH_D"]
    created_times = [
        datetime.now() - timedelta(days=np.random.randint(0, 10),
                                   hours=np.random.randint(0, 23),
                                   minutes=np.random.randint(0, 59))
        for _ in range(n)
    ]
    status_list = []
    processed_times = []
    is_prepacked = []
    for ct in created_times:
        if np.random.rand() > 0.25:
            pt = ct + timedelta(hours=np.random.randint(1, 72))
            processed_times.append(pt)
            status_list.append("processed")
        else:
            processed_times.append(pd.NaT)
            status_list.append("pending")
        is_prepacked.append(np.random.rand() < 0.2)
    df = pd.DataFrame({
        "order_id": [f"ORD{100000+i}" for i in range(n)],
        "warehouse_id": np.random.choice(warehouses, n),
        "status": status_list,
        "created_time": created_times,
        "processed_time": processed_times,
        "is_prepacked": is_prepacked
    })
    return df

def compute_kpis(df, sla_hours=24):
    dfc = df.copy()
    dfc["processing_time"] = (pd.to_datetime(dfc["processed_time"]) - pd.to_datetime(dfc["created_time"])).dt.total_seconds()/3600
    processed = dfc[dfc["status"]=="processed"]
    sla_rate = (processed["processing_time"] < sla_hours).mean()*100 if not processed.empty else 0.0
    avg_time = processed["processing_time"].mean() if not processed.empty else float("nan")
    pending = dfc[dfc["status"]=="pending"].shape[0]
    backlog_per_wh = dfc[dfc["status"]=="pending"].groupby("warehouse_id").size().reset_index(name="pending_orders")
    return {
        "sla_rate": round(sla_rate,2),
        "avg_time": round(avg_time,2) if not pd.isna(avg_time) else None,
        "pending": pending,
        "backlog_per_wh": backlog_per_wh,
        "df_local": dfc
    }

def run_auto_skip_rules(df, rules, now=None):
    """
    rules: dict {
      "enable_prepacked": bool,
      "enable_auto_wh": bool,
      "auto_wh_list": [..],
      "enable_old_order": bool,
      "older_than_hours": int,
      "quick_process_hours": int
    }
    Returns (df_after, applied_entries)
    """
    if now is None:
        now = datetime.now()
    df2 = df.copy()
    applied = []
    cond_pending = df2["status"]=="pending"
    idxs = df2[cond_pending].index
    for i in idxs:
        reasons = []
        do_process = False
        if rules.get("enable_prepacked") and df2.at[i,"is_prepacked"]:
            reasons.append("prepacked")
            do_process = True
        if rules.get("enable_auto_wh") and df2.at[i,"warehouse_id"] in rules.get("auto_wh_list",[]):
            reasons.append("auto_wh")
            do_process = True
        if rules.get("enable_old_order"):
            age_h = (now - pd.to_datetime(df2.at[i,"created_time"])).total_seconds()/3600
            if age_h > rules.get("older_than_hours",72):
                reasons.append("old_order")
                do_process = True
        if do_process:
            prev_status = df2.at[i,"status"]
            df2.at[i,"status"] = "processed"
            # set processed_time: created_time + quick_process_hours OR now
            if rules.get("quick_process_hours") is not None:
                df2.at[i,"processed_time"] = pd.to_datetime(df2.at[i,"created_time"]) + timedelta(hours=rules.get("quick_process_hours"))
            else:
                df2.at[i,"processed_time"] = now
            applied.append({
                "order_id": df2.at[i,"order_id"],
                "warehouse_id": df2.at[i,"warehouse_id"],
                "reason": "|".join(reasons),
                "old_status": prev_status,
                "new_status": "processed",
                "applied_at": now
            })
    return df2, applied

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# -------------------------
# App layout & State
# -------------------------
st.set_page_config(page_title="Radar Warehouse", layout="wide")
st.title("📡 Radar Warehouse")

# Sidebar: Data input & controls
st.sidebar.header("1) Dữ liệu & Thiết lập")
uploaded = st.sidebar.file_uploader("Tải file CSV đơn hàng (các cột: order_id,warehouse_id,status,created_time,processed_time,is_prepacked)", type=["csv"])
use_mock = st.sidebar.checkbox("Sử dụng dữ liệu mô phỏng nếu không upload", value=True)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, parse_dates=["created_time","processed_time"])
        # ensure fields
        if "is_prepacked" not in df.columns:
            df["is_prepacked"] = False
    except Exception as e:
        st.sidebar.error(f"Lỗi khi đọc file: {e}")
        df = generate_mock_data()
else:
    if use_mock:
        df = generate_mock_data(500)
    else:
        st.sidebar.info("Bạn chưa upload file và cũng không chọn dùng mock data.")
        df = generate_mock_data(200)

# Initialize session state for history & audit
if "history_stack" not in st.session_state:
    st.session_state.history_stack = []  # store snapshots before auto-skip for undo/comparison
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# Sidebar: Auto-skip rules
st.sidebar.header("2) Thiết lập Auto-Skip (Quy tắc)")
enable_prepacked = st.sidebar.checkbox("Cho phép auto-skip nếu is_prepacked=True", value=True)
enable_auto_wh = st.sidebar.checkbox("Cho phép auto-skip theo danh sách kho", value=True)
auto_wh_text = st.sidebar.text_input("Danh sách kho auto-ship (phân cách bằng dấu phẩy)", value="WH_A,WH_B")
auto_wh_list = [w.strip() for w in auto_wh_text.split(",") if w.strip()]
enable_old = st.sidebar.checkbox("Cho phép auto-skip đơn hàng cũ hơn X giờ", value=True)
older_than_hours = st.sidebar.number_input("X (giờ) cho quy tắc đơn cũ", min_value=1, value=72)
quick_process_hours = st.sidebar.number_input("Số giờ nhanh để gán processed_time (nếu xử lý auto)", min_value=0, value=1)

enable_auto = st.sidebar.checkbox("Bật Auto-Skip (kích hoạt mới chạy nút)", value=False)
run_auto_btn = st.sidebar.button("▶️ Chạy Auto-Skip bây giờ")

# Sidebar: simulate & export
st.sidebar.header("3) Công cụ khác")
gen_new_btn = st.sidebar.button("➕ Sinh thêm orders mới (simulate real-time)")
download_current = st.sidebar.download_button("📥 Tải dataset hiện tại (CSV)", data=df_to_csv_bytes(df), file_name="orders_current.csv", mime="text/csv")
download_audit = st.sidebar.download_button("📥 Tải Audit Log (CSV)", data=df_to_csv_bytes(pd.DataFrame(st.session_state.audit_log) if st.session_state.audit_log else pd.DataFrame()), file_name="audit_log.csv", mime="text/csv")

# Filters on top
st.subheader("Bộ lọc nhanh")
colf1, colf2, colf3 = st.columns([3,2,2])
with colf1:
    search_order = st.text_input("Tìm kiếm order_id (nhập chuỗi con)", "")
with colf2:
    wh_options = ["All"] + sorted(df["warehouse_id"].unique().tolist())
    sel_wh = st.selectbox("Chọn kho", wh_options, index=0)
with colf3:
    status_options = ["All", "pending", "processed"]
    sel_status = st.selectbox("Chọn trạng thái", status_options, index=0)

# -------------------------
# Simulate incoming orders (mock real-time)
# -------------------------
if gen_new_btn:
    # tạo 20 đơn mới trong 1 batch
    new_n = 20
    new_df = generate_mock_data(new_n)
    # ensure order_id unique: prefix with timestamp
    prefix = datetime.now().strftime("%Y%m%d%H%M%S")
    new_df["order_id"] = [f"{prefix}_{i}" for i in range(new_n)]
    df = pd.concat([new_df, df], ignore_index=True)
    st.success(f"Đã thêm {new_n} đơn mới (mô phỏng incoming orders).")

# Apply filters for display
df_display = df.copy()
if sel_wh != "All":
    df_display = df_display[df_display["warehouse_id"]==sel_wh]
if sel_status != "All":
    df_display = df_display[df_display["status"]==sel_status]
if search_order:
    df_display = df_display[df_display["order_id"].str.contains(search_order, case=False, na=False)]

# -------------------------
# KPI và Visuals
# -------------------------
st.subheader("KPI tổng quan")
kpis = compute_kpis(df, sla_hours=24)
col1, col2, col3, col4 = st.columns(4)
col1.metric("SLA Compliance (%)", f"{kpis['sla_rate']}%")
col2.metric("Thời gian xử lý TB (giờ)", f"{kpis['avg_time'] if kpis['avg_time'] is not None else '-'}")
col3.metric("Số đơn Pending", kpis["pending"])
col4.metric("Số kho (unique)", df["warehouse_id"].nunique())

# Risk alerts
if kpis["sla_rate"] < 95:
    st.error(f"⚠️ SLA thấp: {kpis['sla_rate']}% < 95%")
if kpis["pending"] > 100:
    st.warning(f"⚠️ Backlog lớn: {kpis['pending']} pending orders")

# Visuals
st.subheader("Biểu đồ & Thống kê")
left_col, right_col = st.columns(2)
with left_col:
    backlog_df = kpis["backlog_per_wh"]
    if backlog_df.empty:
        st.info("Không có đơn pending để hiển thị backlog theo kho.")
    else:
        fig_bar = px.bar(backlog_df, x="warehouse_id", y="pending_orders", labels={"warehouse_id":"Warehouse","pending_orders":"Pending Orders"}, title="Backlog theo kho")
        st.plotly_chart(fig_bar, use_container_width=True)
with right_col:
    proc_df = kpis["df_local"][kpis["df_local"]["status"]=="processed"].copy()
    if not proc_df.empty:
        proc_df["processed_date"] = pd.to_datetime(proc_df["processed_time"]).dt.date
        trend = proc_df.groupby("processed_date")["processing_time"].mean().reset_index()
        fig_line = px.line(trend, x="processed_date", y="processing_time", markers=True, labels={"processed_date":"Date","processing_time":"Avg Processing Time (h)"}, title="Avg Processing Time per Day")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Chưa có đơn processed để hiển thị trend.")

# -------------------------
# Detailed table + export selection
# -------------------------
st.subheader("Bảng đơn hàng (chi tiết) — có thể sort/search")
# show interactive slice (limit)
st.dataframe(df_display.sort_values(by="created_time", ascending=False).reset_index(drop=True).head(500))

# download filtered view
csv_bytes = df_display.to_csv(index=False).encode("utf-8")
st.subheader("📜 Audit Log – Danh sách đơn hàng được Auto-Skip")

if st.session_state.audit_log:
    df_log = pd.DataFrame(st.session_state.audit_log)
    st.dataframe(df_log)

    # Thêm nút download, với key để tránh trùng ID
    st.download_button(
        "📥 Tải Audit Log (CSV)",
        data=df_log.to_csv(index=False).encode("utf-8"),
        file_name="audit_log.csv",
        mime="text/csv",
        key="download_audit_log"
    )
else:
    st.info("Chưa có đơn hàng nào được Auto-Skip.")

# -------------------------
# Auto-skip execution & comparison
# -------------------------
st.subheader("Auto-Skip — Thực thi & So sánh Before/After")

# prepare rules dict
rules = {
    "enable_prepacked": enable_prepacked,
    "enable_auto_wh": enable_auto_wh,
    "auto_wh_list": auto_wh_list,
    "enable_old_order": enable_old,
    "older_than_hours": older_than_hours,
    "quick_process_hours": quick_process_hours
}

if run_auto_btn:
    if not enable_auto:
        st.warning("Bạn chưa tick 'Bật Auto-Skip' trong sidebar. Tick để cho phép chạy Auto-Skip.")
    else:
        # snapshot before
        before_snapshot = df.copy()
        st.session_state.history_stack.append(before_snapshot)
        # run auto-skip
        df_after, applied_entries = run_auto_skip_rules(df, rules)
        # update global df
        df = df_after
        # save audit entries
        if applied_entries:
            st.session_state.audit_log.extend(applied_entries)
        st.success(f"Auto-Skip đã chạy. Tổng {len(applied_entries)} đơn được auto-skip.")
        # compute comparison
        before_kpis = compute_kpis(before_snapshot, sla_hours=24)
        after_kpis = compute_kpis(df, sla_hours=24)
        # show comparison
        st.markdown("**So sánh Before / After (sau Auto-Skip)**")
        comp_cols = st.columns(4)
        comp_cols[0].metric("SLA Before", f"{before_kpis['sla_rate']}%")
        comp_cols[1].metric("SLA After", f"{after_kpis['sla_rate']}%")
        comp_cols[2].metric("Pending Before", before_kpis["pending"])
        comp_cols[3].metric("Pending After", after_kpis["pending"])
        # show delta
        delta_pending = before_kpis["pending"] - after_kpis["pending"]
        st.info(f"🟢 Số đơn được xử lý bằng Auto-Skip: {delta_pending}")

# Undo last auto-skip
if st.button("↩️ Hoàn tác lần Auto-Skip gần nhất"):
    if st.session_state.history_stack:
        df = st.session_state.history_stack.pop()
        st.success("Đã hoàn tác: phục hồi snapshot trước Auto-Skip.")
    else:
        st.warning("Không có lịch sử để hoàn tác.")

# Show audit log
st.subheader("Audit Log (lịch sử auto-skip)")
if st.session_state.audit_log:
    df_log = pd.DataFrame(st.session_state.audit_log)
    # format time
    df_log["applied_at"] = pd.to_datetime(df_log["applied_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(df_log.sort_values("applied_at", ascending=False).reset_index(drop=True).head(500))
    st.download_button("📥 Tải Audit Log (CSV)", data=df_to_csv_bytes(df_log), file_name="audit_log.csv", mime="text/csv")
else:
    st.info("Chưa có bản ghi auto-skip nào.")

# Persist updated df to session (so subsequent actions use updated df)
st.session_state["current_data"] = df

# -------------------------
# Notes & next steps
# -------------------------
st.markdown("---")
st.markdown(
"""
**Gợi ý dùng khi demo trước team:**
- Trình bày KPI _Before_ và _After_ để chứng minh auto-skip giảm backlog ngay lập tức.
- Mở Audit Log để minh bạch: ai/đơn nào bị auto-skip và lý do (prepacked/auto_wh/old_order).
- Nêu rủi ro & cách mitigation: chỉ cho phép auto-skip với SKU low-risk hoặc cần approval workflow cho các SKU giá trị.
- Kế tiếp: tích hợp API WMS/DB thật, add role-based access control, add monitoring/alerting real-time (email/Slack).
"""
)

# End of file
