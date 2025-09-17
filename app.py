import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from streamlit_calendar import calendar
import os

# Cấu hình page
st.set_page_config(
    page_title="Time Off Dashboard", 
    page_icon="🏖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class EmployeeManager:
    """Class để quản lý thông tin nhân viên và mapping username -> name"""
    
    def __init__(self, account_token: str):
        self.account_token = account_token
        self.request_timeout = 30
        self.username_to_name_map = {}
        self._load_employee_mapping()
    
    def _make_request(self, url: str, data: Dict, description: str = "") -> requests.Response:
        """Thực hiện HTTP request với error handling"""
        try:
            response = requests.post(url, data=data, timeout=self.request_timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            st.error(f"Error {description}: {e}")
            raise
    
    def _load_employee_mapping(self):
        """Tải mapping username -> name từ account API"""
        url = "https://account.base.vn/extapi/v1/group/get"
        data = {"access_token": self.account_token, "path": "aplus"}
        
        try:
            response = self._make_request(url, data, "fetching account members")
            response_data = response.json()
            
            members = response_data.get('group', {}).get('members', [])
            
            # Tạo mapping username -> name
            self.username_to_name_map = {
                m.get('username', ''): m.get('name', '') 
                for m in members 
                if m.get('username') and m.get('name')
            }
            
        except Exception as e:
            st.error(f"Lỗi khi lấy danh sách nhân viên: {e}")
            self.username_to_name_map = {}
    
    def get_name_by_username(self, username: str) -> str:
        """Lấy name từ username, fallback về username nếu không tìm thấy"""
        if not username:
            return ''
        return self.username_to_name_map.get(username, username)

class TimeoffProcessor:
    """Class để xử lý dữ liệu timeoff và thay thế username bằng name"""
    
    def __init__(self, timeoff_token: str, account_token: str):
        self.timeoff_token = timeoff_token
        self.employee_manager = EmployeeManager(account_token)
        
    def get_base_timeoff_data(self):
        """Lấy dữ liệu từ Base Timeoff API"""
        url = "https://timeoff.base.vn/extapi/v1/timeoff/list"
        
        payload = f'access_token={self.timeoff_token}'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(url, headers=headers, data=payload)
        return response.json()
    
    def extract_form_data(self, form_list):
        """Extract dữ liệu từ form fields"""
        form_data = {}
        for form_item in form_list:
            if form_item.get('name') and form_item.get('value'):
                form_data[form_item['name']] = form_item['value']
        return form_data
    
    def clean_vietnamese_text(self, text):
        """Clean Vietnamese text for column names"""
        import unicodedata
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        text = text.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
        return text
    
    def convert_timestamp_to_date(self, timestamp):
        """Chuyển timestamp thành datetime"""
        if timestamp and timestamp != '0':
            try:
                return datetime.fromtimestamp(int(timestamp))
            except:
                return None
        return None
    
    def convert_approvals_to_names(self, approvals: List[str]) -> str:
        """Chuyển đổi danh sách approval usernames thành names"""
        if not approvals:
            return ''
        
        approval_names = []
        for username in approvals:
            name = self.employee_manager.get_name_by_username(username)
            approval_names.append(name)
        
        return ', '.join(approval_names)
    
    def create_ly_do_column_and_cleanup(self, df):
        """Tạo cột 'ly_do' từ các cột có sẵn theo thứ tự ưu tiên"""
        if df.empty:
            return df
            
        df_copy = df.copy()
        df_copy['ly_do'] = ''
        
        priority_columns = ['ly_do_xin_nghi_phep', 'ly_do_xin_nghi_chinh', 'ly_do_xin_nghi']
        
        for col in priority_columns:
            if col in df_copy.columns:
                mask = (
                    (df_copy['ly_do'] == '') & 
                    (df_copy[col].notna()) & 
                    (df_copy[col].astype(str).str.strip() != '')
                )
                df_copy.loc[mask, 'ly_do'] = df_copy.loc[mask, col].astype(str).str.strip()
        
        columns_to_drop = [col for col in priority_columns if col in df_copy.columns]
        if columns_to_drop:
            df_copy = df_copy.drop(columns=columns_to_drop)
        
        return df_copy
    
    def extract_timeoff_to_dataframe(self, api_response):
        """Extract các trường quan trọng từ Base Timeoff API response thành DataFrame với name mapping"""
        timeoffs_data = []
        
        if 'timeoffs' in api_response:
            for timeoff in api_response['timeoffs']:
                form_data = self.extract_form_data(timeoff.get('form', []))
                
                approvals = timeoff.get('approvals', [])
                approval_names = self.convert_approvals_to_names(approvals)
                
                total_shifts = len(timeoff.get('shifts', []))
                
                total_leave_days = 0
                for shift_day in timeoff.get('shifts', []):
                    for shift in shift_day.get('shifts', []):
                        if 'num_leave' in shift:
                            total_leave_days += float(shift.get('num_leave', 0))
                
                final_approver_username = ''
                final_approver_name = ''
                if timeoff.get('data', {}).get('final_approved'):
                    final_approver_username = timeoff['data']['final_approved'].get('username', '')
                    final_approver_name = self.employee_manager.get_name_by_username(final_approver_username)
                
                username = timeoff.get('username', '')
                employee_name = self.employee_manager.get_name_by_username(username)
                
                timeoff_record = {
                    'id': timeoff.get('id'),
                    'employee_name': employee_name,
                    'username': username,
                    'state': timeoff.get('state'),
                    'metatype': timeoff.get('metatype'),
                    'paid_timeoff': timeoff.get('paid_timeoff'),
                    'start_date': self.convert_timestamp_to_date(timeoff.get('start_date')),
                    'end_date': self.convert_timestamp_to_date(timeoff.get('end_date')),
                    'total_leave_days': total_leave_days,
                    'total_shifts': total_shifts,
                    'approvals': approval_names,
                    'final_approver': final_approver_name,
                    'workflow': timeoff.get('workflow'),
                    'created_time': self.convert_timestamp_to_date(timeoff.get('since')),
                    'last_update': self.convert_timestamp_to_date(timeoff.get('last_update')),
                }
                
                # Add form data fields
                column_mapping = {
                    'Lý do xin nghỉ phép': 'ly_do_xin_nghi_phep',
                    'Lý do xin nghỉ': 'ly_do_xin_nghi',  
                    'Lý do': 'ly_do_xin_nghi',
                    'Ghi chú': 'ghi_chu',
                    'Lý do cá nhân': 'ly_do_ca_nhan',
                    'Bận việc cá nhân': 'ban_viec_ca_nhan',
                    'Việc riêng': 'viec_rieng'
                }
                
                for key, value in form_data.items():
                    if key in column_mapping:
                        clean_key = column_mapping[key]
                    else:
                        clean_key = self.clean_vietnamese_text(key)
                    timeoff_record[clean_key] = value
                
                timeoff_record['ly_do_xin_nghi_chinh'] = (
                    form_data.get('Lý do xin nghỉ phép', '') or 
                    form_data.get('Lý do xin nghỉ', '') or
                    form_data.get('Lý do', '') or
                    form_data.get('Lý do cá nhân', '') or
                    form_data.get('Bận việc cá nhân', '') or
                    form_data.get('Việc riêng', '')
                )
                
                timeoffs_data.append(timeoff_record)
        
        df = pd.DataFrame(timeoffs_data)
        
        if not df.empty and 'created_time' in df.columns:
            df = df.sort_values('created_time', ascending=False)
        
        df = self.create_ly_do_column_and_cleanup(df)
        
        return df

# Cache dữ liệu để tránh gọi API liên tục
@st.cache_data(ttl=300)  # Cache 5 phút
def load_timeoff_data():
    """Load dữ liệu timeoff với caching"""
    # Lấy tokens từ environment variables
    timeoff_token = os.getenv('TIMEOFF_TOKEN')
    account_token = os.getenv('ACCOUNT_TOKEN')
    
    processor = TimeoffProcessor(timeoff_token, account_token)
    
    try:
        api_data = processor.get_base_timeoff_data()
        df = processor.extract_timeoff_to_dataframe(api_data)
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return pd.DataFrame()

def get_state_info():
    """Trả về thông tin về các trạng thái và màu sắc"""
    return {
        'approved': {'color': '#28a745', 'icon': '✅', 'label': 'Đã duyệt'},
        'pending': {'color': '#ffc107', 'icon': '⏳', 'label': 'Chờ duyệt'},
        'rejected': {'color': '#dc3545', 'icon': '❌', 'label': 'Từ chối'},
        'cancelled': {'color': '#6c757d', 'icon': '⏹️', 'label': 'Đã hủy'},
        'draft': {'color': '#17a2b8', 'icon': '📝', 'label': 'Nháp'}
    }

def get_metatype_info():
    """Trả về thông tin về các loại nghỉ phép và màu sắc"""
    return {
        'annual': {'color': '#28a745', 'icon': '🏖️', 'label': 'Nghỉ phép năm'},
        'sick': {'color': '#fd7e14', 'icon': '🤒', 'label': 'Nghỉ ốm'},
        'unpaid': {'color': '#dc3545', 'icon': '💸', 'label': 'Nghỉ không lương'},
        'personal': {'color': '#6f42c1', 'icon': '👤', 'label': 'Nghỉ cá nhân'},
        'outside': {'color': '#20c997', 'icon': '🏢', 'label': 'Công tác ngoài'},
        'maternity': {'color': '#e83e8c', 'icon': '👶', 'label': 'Nghỉ thai sản'},
        'wedding': {'color': '#fd7e14', 'icon': '💒', 'label': 'Nghỉ cưới'},
        'funeral': {'color': '#6c757d', 'icon': '🕊️', 'label': 'Nghỉ tang'}
    }

def convert_df_to_calendar_events(df):
    """Chuyển DataFrame thành format events cho calendar với tối ưu hiển thị text"""
    events = []
    
    if df.empty:
        return events
    
    state_info = get_state_info()
    metatype_info = get_metatype_info()
    
    for _, row in df.iterrows():
        if pd.notna(row['start_date']) and pd.notna(row['end_date']):
            # Determine color based on state and metatype
            if row['state'] == 'approved':
                color = metatype_info.get(row['metatype'], {}).get('color', '#28a745')
                icon = metatype_info.get(row['metatype'], {}).get('icon', '📅')
            else:
                color = state_info.get(row['state'], {}).get('color', '#007bff')
                icon = state_info.get(row['state'], {}).get('icon', '📅')
            
            # Tối ưu title để tránh bị cắt
            # Rút ngắn tên nhân viên nếu quá dài
            employee_name = row['employee_name']
            if len(employee_name) > 15:
                name_parts = employee_name.split()
                if len(name_parts) > 1:
                    # Lấy tên và chữ cái đầu họ
                    employee_name = f"{name_parts[-1]} {name_parts[0][0]}."
                else:
                    employee_name = employee_name[:12] + "..."
            
            # Format title với xuống dòng
            title = f"{icon} {employee_name}"
            
            # Thêm thông tin lý do hoặc loại nghỉ với xuống dòng
            if row['ly_do'] and row['ly_do'] != '':
                reason = row['ly_do']
                if len(reason) > 20:
                    reason = reason[:17] + "..."
                title += f"\n{reason}"
            else:
                metatype_label = metatype_info.get(row['metatype'], {}).get('label', row['metatype'].title())
                if len(metatype_label) > 20:
                    metatype_label = metatype_label[:17] + "..."
                title += f"\n{metatype_label}"
            
            # Thêm số ngày với xuống dòng
            if row['total_leave_days'] > 0:
                title += f"\n({row['total_leave_days']} ngày)"
            
            # Create comprehensive event
            event = {
                "title": title,
                "start": row['start_date'].strftime('%Y-%m-%d'),
                "end": (row['end_date'] + timedelta(days=1)).strftime('%Y-%m-%d'),
                "color": color,
                "borderColor": color,
                "textColor": "#ffffff",
                "extendedProps": {
                    "id": row['id'],
                    "employee": row['employee_name'],
                    "state": row['state'],
                    "metatype": row['metatype'],
                    "days": row['total_leave_days'],
                    "reason": row['ly_do'],
                    "approver": row['final_approver'],
                    "created_time": row['created_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['created_time']) else 'N/A',
                    "last_update": row['last_update'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['last_update']) else 'N/A',
                    "paid": row['paid_timeoff'] if 'paid_timeoff' in row else False
                },
                "display": "block"
            }
            events.append(event)
    
    return events

def display_calendar_legend():
    """Hiển thị chú thích màu sắc cho calendar"""
    state_info = get_state_info()
    metatype_info = get_metatype_info()
    
    st.markdown("#### 📋 Chú thích")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trạng thái:**")
        for state, info in state_info.items():
            st.markdown(f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                       f"<div style='width: 20px; height: 20px; background-color: {info['color']}; "
                       f"border-radius: 3px; margin-right: 10px;'></div>"
                       f"<span>{info['icon']} {info['label']}</span></div>", 
                       unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Loại nghỉ phép:**")
        for metatype, info in metatype_info.items():
            st.markdown(f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                       f"<div style='width: 20px; height: 20px; background-color: {info['color']}; "
                       f"border-radius: 3px; margin-right: 10px;'></div>"
                       f"<span>{info['icon']} {info['label']}</span></div>", 
                       unsafe_allow_html=True)

def display_event_details(event_data):
    """Hiển thị chi tiết event khi click"""
    if not event_data:
        return
    
    props = event_data.get('extendedProps', {})
    state_info = get_state_info()
    metatype_info = get_metatype_info()
    
    # Get state and metatype info
    state = props.get('state', '')
    metatype = props.get('metatype', '')
    state_display = state_info.get(state, {})
    metatype_display = metatype_info.get(metatype, {})
    
    st.markdown("### 📋 Chi tiết yêu cầu nghỉ phép")
    
    # Create a nice card layout
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; margin: 10px 0;">
        <h4 style="margin: 0; color: white;">👤 {props.get('employee', 'N/A')}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Thông tin thời gian:**")
        st.info(f"**Từ ngày:** {event_data.get('start', 'N/A')}\n"
                f"**Đến ngày:** {event_data.get('end', 'N/A')}\n"
                f"**Số ngày:** {props.get('days', 0)} ngày")
        
        st.markdown("**📊 Trạng thái & Loại:**")
        status_color = state_display.get('color', '#007bff')
        metatype_color = metatype_display.get('color', '#007bff')
        
        st.markdown(f"""
        <div style="margin: 10px 0;">
            <span style="background-color: {status_color}; color: white; padding: 5px 10px; 
                         border-radius: 15px; font-size: 14px;">
                {state_display.get('icon', '')} {state_display.get('label', state)}
            </span>
        </div>
        <div style="margin: 10px 0;">
            <span style="background-color: {metatype_color}; color: white; padding: 5px 10px; 
                         border-radius: 15px; font-size: 14px;">
                {metatype_display.get('icon', '')} {metatype_display.get('label', metatype)}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**📝 Thông tin chi tiết:**")
        reason = props.get('reason', 'Không có thông tin')
        if reason and reason.strip():
            st.success(f"**Lý do:** {reason}")
        else:
            st.info("**Lý do:** Không có thông tin")
        
        approver = props.get('approver', 'N/A')
        if approver and approver.strip():
            st.success(f"**Người duyệt:** {approver}")
        
        # Additional info
        paid_status = "Có lương" if props.get('paid', False) else "Không lương"
        st.info(f"**Loại:** {paid_status}")
    
    # Timeline info
    st.markdown("**⏰ Thời gian xử lý:**")
    col3, col4 = st.columns(2)
    with col3:
        st.text(f"Tạo: {props.get('created_time', 'N/A')}")
    with col4:
        st.text(f"Cập nhật: {props.get('last_update', 'N/A')}")

def main():
    """Main dashboard"""
    
    # Custom CSS cho calendar và hiển thị text tốt hơn
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .calendar-container {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        margin: 5px 0;
        padding: 5px;
        background: #f8f9fa;
        border-radius: 5px;
    }
    
    /* Tối ưu CSS cho calendar events */
    .fc-event {
        font-size: 11px !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 2px 4px !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
        transition: all 0.2s ease !important;
        white-space: pre-line !important;
        line-height: 1.2 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        min-height: 20px !important;
    }
    
    .fc-event:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        z-index: 1000 !important;
    }
    
    .fc-event-title {
        font-weight: 600 !important;
        white-space: pre-line !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        line-height: 1.1 !important;
    }
    
    .fc-daygrid-event {
        margin: 1px 2px !important;
        min-height: 22px !important;
    }
    
    .fc-daygrid-event-harness {
        margin-bottom: 2px !important;
    }
    
    /* Tăng chiều cao của calendar cells */
    .fc-daygrid-day {
        min-height: 80px !important;
    }
    
    .fc-daygrid-day-frame {
        min-height: 80px !important;
    }
    
    .fc-daygrid-day-events {
        margin-bottom: 2px !important;
    }
    
    .fc-button-primary {
        background-color: #667eea !important;
        border-color: #667eea !important;
    }
    
    .fc-button-primary:hover {
        background-color: #764ba2 !important;
        border-color: #764ba2 !important;
    }
    
    .fc-today-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .fc-header-toolbar {
        margin-bottom: 1em !important;
        padding: 10px !important;
        background: #f8f9fa !important;
        border-radius: 8px !important;
    }
    
    .fc-col-header-cell {
        background: #f8f9fa !important;
        font-weight: 600 !important;
    }
    
    .fc-day-today {
        background-color: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Responsive font size */
    @media (max-width: 768px) {
        .fc-event {
            font-size: 10px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with gradient
    st.markdown("""
    <div class="main-header">
        <h1>🏖️ Time Off Management Dashboard</h1>
        <p>Quản lý và theo dõi yêu cầu nghỉ phép một cách hiệu quả</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Đang tải dữ liệu..."):
        df = load_timeoff_data()
    
    if df.empty:
        st.error("❌ Không thể tải dữ liệu timeoff")
        return
    
    # Sidebar filters with better styling
    st.sidebar.markdown("## 🔍 Bộ lọc dữ liệu")
    
    # Date range filter
    if not df.empty and 'start_date' in df.columns:
        min_date = df['start_date'].min().date() if pd.notna(df['start_date'].min()) else datetime.now().date()
        max_date = df['start_date'].max().date() if pd.notna(df['start_date'].max()) else datetime.now().date()
        
        date_range = st.sidebar.date_input(
            "📅 Khoảng thời gian",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Employee filter
    employees = ['Tất cả'] + sorted(df['employee_name'].unique().tolist())
    selected_employees = st.sidebar.multiselect(
        "👥 Nhân viên", 
        employees, 
        default=['Tất cả']
    )
    
    # State filter with icons
    state_info = get_state_info()
    state_options = ['Tất cả'] + [f"{info['icon']} {info['label']}" for state, info in state_info.items()]
    selected_states_display = st.sidebar.multiselect(
        "📊 Trạng thái",
        state_options,
        default=['Tất cả']
    )
    
    # Convert back to original state values
    selected_states = []
    if 'Tất cả' not in selected_states_display:
        for display in selected_states_display:
            for state, info in state_info.items():
                if f"{info['icon']} {info['label']}" == display:
                    selected_states.append(state)
    else:
        selected_states = ['Tất cả']
    
    # Metatype filter with icons
    metatype_info = get_metatype_info()
    metatype_options = ['Tất cả'] + [f"{info['icon']} {info['label']}" for meta, info in metatype_info.items()]
    selected_metatypes_display = st.sidebar.multiselect(
        "📋 Loại nghỉ phép",
        metatype_options,
        default=['Tất cả']
    )
    
    # Convert back to original metatype values
    selected_metatypes = []
    if 'Tất cả' not in selected_metatypes_display:
        for display in selected_metatypes_display:
            for metatype, info in metatype_info.items():
                if f"{info['icon']} {info['label']}" == display:
                    selected_metatypes.append(metatype)
    else:
        selected_metatypes = ['Tất cả']
    
    # Apply filters
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['start_date'].dt.date >= date_range[0]) & 
            (filtered_df['start_date'].dt.date <= date_range[1])
        ]
    
    # Employee filter
    if 'Tất cả' not in selected_employees:
        filtered_df = filtered_df[filtered_df['employee_name'].isin(selected_employees)]
    
    # State filter
    if 'Tất cả' not in selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
    
    # Metatype filter
    if 'Tất cả' not in selected_metatypes:
        filtered_df = filtered_df[filtered_df['metatype'].isin(selected_metatypes)]
    
    # Summary metrics with better styling
    st.markdown("### 📊 Tổng quan")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Tổng số yêu cầu", 
            value=len(filtered_df),
            delta=f"{len(filtered_df)} requests"
        )
    
    with col2:
        approved_count = len(filtered_df[filtered_df['state'] == 'approved'])
        approval_rate = (approved_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric(
            label="✅ Đã duyệt", 
            value=approved_count,
            delta=f"{approval_rate:.1f}% tỷ lệ duyệt"
        )
    
    with col3:
        pending_count = len(filtered_df[filtered_df['state'] == 'pending'])
        st.metric(
            label="⏳ Chờ duyệt", 
            value=pending_count,
            delta=f"{pending_count} pending"
        )
    
    with col4:
        total_days = filtered_df['total_leave_days'].sum()
        avg_days = total_days / len(filtered_df) if len(filtered_df) > 0 else 0
        st.metric(
            label="📅 Tổng số ngày nghỉ", 
            value=f"{total_days:.1f}",
            delta=f"{avg_days:.1f} ngày/yêu cầu"
        )
    
    # Tabs with improved styling
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Calendar View", 
        "📊 Analytics", 
        "📋 Data Table", 
        "⚙️ Settings"
    ])
    
    with tab1:
        st.markdown("### 📅 Lịch Time Off")
        
        # Calendar controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Calendar mode selection with better options
            mode_options = {
                "dayGridMonth": "📅 Tháng",
                "dayGridWeek": "📅 Tuần", 
                "timeGridWeek": "⏰ Tuần (giờ)",
                "listMonth": "📋 Danh sách tháng"
            }
            
            selected_mode_display = st.selectbox(
                "Chế độ xem:",
                list(mode_options.values()),
                index=0
            )
            
            # Get actual mode value
            mode = [k for k, v in mode_options.items() if v == selected_mode_display][0]
        
        with col2:
            st.markdown("**🎨 Tùy chọn hiển thị**")
            show_legend = st.checkbox("Hiển thị chú thích", value=True)
            show_weekend = st.checkbox("Hiển thị cuối tuần", value=True)
        
        # Convert data to events
        events = convert_df_to_calendar_events(filtered_df)
        
        # Enhanced calendar options với tối ưu hiển thị
        calendar_options = {
            "editable": False,
            "navLinks": True,
            "selectable": False,
            "dayMaxEvents": 5,  # Tăng số events tối đa
            "moreLinkClick": "popover",
            "eventDisplay": "block",
            "displayEventTime": False,
            "weekends": show_weekend,
            "headerToolbar": {
                "left": "today prev,next",
                "center": "title", 
                "right": "dayGridMonth,dayGridWeek,timeGridWeek,listMonth"
            },
            "footerToolbar": {
                "left": "",
                "center": "",
                "right": ""
            },
            "initialView": mode,
            "height": 750,  # Tăng chiều cao
            "eventMouseEnter": True,
            "eventMouseLeave": True,
            "locale": "vi",
            "buttonText": {
                "today": "Hôm nay",
                "month": "Tháng",
                "week": "Tuần", 
                "day": "Ngày",
                "list": "Danh sách"
            },
            "dayMaxEventRows": 4,  # Giới hạn số dòng events
            "moreLinkText": "thêm",  # Text cho link "more"
            "eventMinHeight": 20,  # Chiều cao tối thiểu của event
        }
        
        # Display calendar
        if events:
            st.markdown('<div class="calendar-container">', unsafe_allow_html=True)
            
            calendar_state = calendar(
                events=events,
                options=calendar_options,
                key="timeoff_calendar"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show selected event details in a nice format
            if calendar_state.get("eventClick"):
                event_data = calendar_state["eventClick"]["event"]
                display_event_details(event_data)
                
        else:
            st.info("📅 Không có dữ liệu time off trong khoảng thời gian được chọn")
            st.markdown("**Gợi ý:** Thử điều chỉnh bộ lọc để xem thêm dữ liệu")
        
        # Hiển thị chú thích nếu được chọn
        if show_legend:
            display_calendar_legend()
    
    with tab2:
        st.subheader("📊 Phân tích dữ liệu")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Chart by state với màu sắc tùy chỉnh
                state_counts = filtered_df['state'].value_counts()
                colors = [get_state_info().get(state, {}).get('color', '#007bff') for state in state_counts.index]
                
                fig1 = px.pie(
                    values=state_counts.values, 
                    names=[get_state_info().get(state, {}).get('label', state) for state in state_counts.index],
                    title="🎯 Phân bố theo trạng thái",
                    color_discrete_sequence=colors
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                fig1.update_layout(showlegend=True, height=400)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Top employees
                top_employees = filtered_df['employee_name'].value_counts().head(10)
                fig3 = px.bar(
                    x=top_employees.values,
                    y=top_employees.index,
                    orientation='h',
                    title="👥 Top 10 nhân viên có nhiều yêu cầu nhất",
                    labels={'x': 'Số yêu cầu', 'y': 'Nhân viên'},
                    color=top_employees.values,
                    color_continuous_scale="viridis"
                )
                fig3.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Chart by metatype với màu sắc tùy chỉnh
                metatype_counts = filtered_df['metatype'].value_counts()
                colors = [get_metatype_info().get(meta, {}).get('color', '#007bff') for meta in metatype_counts.index]
                
                fig2 = px.bar(
                    x=[get_metatype_info().get(meta, {}).get('label', meta) for meta in metatype_counts.index],
                    y=metatype_counts.values,
                    title="📋 Phân bố theo loại nghỉ phép",
                    labels={'x': 'Loại', 'y': 'Số lượng'},
                    color=metatype_counts.values,
                    color_discrete_sequence=colors
                )
                fig2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Timeline
                if 'start_date' in filtered_df.columns:
                    monthly_data = filtered_df.groupby(
                        filtered_df['start_date'].dt.to_period('M')
                    ).size().reset_index()
                    monthly_data.columns = ['Tháng', 'Số yêu cầu']
                    monthly_data['Tháng'] = monthly_data['Tháng'].astype(str)
                    
                    fig4 = px.line(
                        monthly_data, 
                        x='Tháng', 
                        y='Số yêu cầu',
                        title="📈 Xu hướng theo thời gian",
                        markers=True
                    )
                    fig4.update_traces(line=dict(width=3), marker=dict(size=8))
                    fig4.update_layout(height=400)
                    st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        st.subheader("📋 Bảng dữ liệu")
        
        # Display options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            show_all_columns = st.checkbox("Hiển thị tất cả cột", False)
        with col3:
            items_per_page = st.selectbox("Số dòng/trang", [10, 25, 50, 100], index=1)
        
        if not filtered_df.empty:
            if show_all_columns:
                display_df = filtered_df
            else:
                # Select important columns
                important_cols = [
                    'employee_name', 'state', 'metatype', 'start_date', 'end_date', 
                    'total_leave_days', 'ly_do', 'final_approver'
                ]
                available_cols = [col for col in important_cols if col in filtered_df.columns]
                display_df = filtered_df[available_cols]
            
            # Format dates
            if 'start_date' in display_df.columns:
                display_df['start_date'] = pd.to_datetime(display_df['start_date']).dt.strftime('%Y-%m-%d')
            if 'end_date' in display_df.columns:
                display_df['end_date'] = pd.to_datetime(display_df['end_date']).dt.strftime('%Y-%m-%d')
            
            # Pagination
            total_rows = len(display_df)
            total_pages = (total_rows + items_per_page - 1) // items_per_page
            
            if total_pages > 1:
                page = st.selectbox(f"Trang (1-{total_pages})", range(1, total_pages + 1))
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                display_df_page = display_df.iloc[start_idx:end_idx]
            else:
                display_df_page = display_df
            
            st.dataframe(
                display_df_page,
                use_container_width=True,
                hide_index=True
            )
            
            # Download section
            st.markdown("### 📥 Tải xuống dữ liệu")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Tải xuống dữ liệu đã lọc (CSV)",
                    data=csv,
                    file_name=f"filtered_timeoff_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                # JSON download
                json_data = filtered_df.to_json(orient='records', date_format='iso', force_ascii=False)
                st.download_button(
                    label="📥 Tải xuống dữ liệu đã lọc (JSON)",
                    data=json_data,
                    file_name=f"filtered_timeoff_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        else:
            st.info("📭 Không có dữ liệu phù hợp với bộ lọc")
    
    with tab4:
        st.subheader("⚙️ Cài đặt hệ thống")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Thông tin hệ thống")
            info_data = {
                "Tổng số records": len(df),
                "Số nhân viên": df['employee_name'].nunique(),
                "Records sau lọc": len(filtered_df),
                "Khoảng thời gian": f"{df['start_date'].min().strftime('%Y-%m-%d') if pd.notna(df['start_date'].min()) else 'N/A'} → {df['start_date'].max().strftime('%Y-%m-%d') if pd.notna(df['start_date'].max()) else 'N/A'}"
            }
            
            for label, value in info_data.items():
                st.metric(label, value)
        
        with col2:
            st.markdown("#### 🔧 Công cụ quản lý")
            
            # Refresh button
            if st.button("🔄 Làm mới dữ liệu", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            st.markdown("---")
            
            # Export full data
            st.markdown("**📤 Xuất toàn bộ dữ liệu:**")
            full_csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải xuống toàn bộ dữ liệu (CSV)",
                data=full_csv,
                file_name=f"full_timeoff_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # System stats
            st.markdown("---")
            st.markdown("**⚡ Trạng thái hệ thống:**")
            st.success("✅ API Connection: Active")
            st.success("✅ Data Cache: Active")
            st.success("✅ Environment Variables: Loaded")
            st.info(f"🕒 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
