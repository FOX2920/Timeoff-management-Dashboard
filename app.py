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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata
import pytz

# Cấu hình page
st.set_page_config(
    page_title="Time Off Dashboard", 
    page_icon="🏖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class ReasonClassifier:
    """Class để phân loại lý do nghỉ bằng cosine similarity"""
    
    def __init__(self):
        # Định nghĩa categories và từ khóa đại diện theo yêu cầu mới
        self.categories = {
            'annual_leave': {
                'keywords': [
                    'phép năm', 'nghỉ phép', 'annual leave', 'vacation', 'holiday',
                    'du lịch', 'đi chơi', 'nghỉ mát', 'resort', 'biển', 'núi',
                    'về quê', 'thăm quê', 'nghỉ dưỡng', 'thư giãn', 'relax',
                    'break', 'nghỉ ngơi', 'rest', 'phục hồi', 'tái tạo năng lượng',
                    'đi du lịch', 'travel', 'trip', 'picnic', 'tour', 'khám phá',
                    'nghỉ lễ', 'long weekend', 'nghỉ cuối tuần', 'staycation'
                ],
                'color': '#28a745',  # Xanh lá
                'icon': '🏖️',
                'label': 'Phép năm'
            },
            'personal': {
                'keywords': [
                    'cá nhân', 'việc riêng', 'bận việc cá nhân', 'công việc cá nhân',
                    'giải quyết việc', 'làm việc cá nhân', 'việc tư', 'tự do',
                    'mua sắm', 'đi ngân hàng', 'làm giấy tờ', 'visa', 'hộ chiếu',
                    'sửa nhà', 'chuyển nhà', 'dọn nhà', 'việc nhà'
                ],
                'color': '#6f42c1',  # Tím
                'icon': '👤',
                'label': 'Cá nhân'
            },
            'remote': {
                'keywords': [
                    'remote', 'work from home', 'wfh', 'làm việc từ xa','outside',
                    'làm việc tại nhà', 'online', 'từ xa', 'không đến công ty',
                    'ở nhà làm việc', 'home office', 'telecommuting', 'virtual work'
                ],
                'color': '#17a2b8',  # Xanh dương nhạt
                'icon': '💻',
                'label': 'Remote'
            },
            'business': {
                'keywords': [
                    'công tác', 'business trip', 'công việc', 'meeting', 'họp',
                    'hội nghị', 'đào tạo', 'khóa học', 'seminar', 'conference',
                    'gặp khách hàng', 'partner', 'đối tác', 'dự án', 'project',
                    'ra ngoài công tác', 'đi công tác', 'business'
                ],
                'color': '#fd7e14',  # Cam
                'icon': '💼',
                'label': 'Công tác'
            },
            'sick': {
                'keywords': [
                    'ốm', 'bệnh', 'đau', 'sốt', 'cảm', 'ho', 'khám bệnh', 'chữa bệnh',
                    'bác sĩ', 'bệnh viện', 'phòng khám', 'điều trị', 'thuốc', 'y tế',
                    'sức khỏe', 'không khỏe', 'mệt', 'kiệt sức', 'stress', 'lo âu',
                    'sick', 'ill', 'medical', 'doctor', 'hospital', 'fever', 'cold',
                    'đau đầu', 'đau bụng', 'đau răng', 'cúm', 'viêm họng', 'ho khan',
                    'sốt cao', 'sốt nhẹ', 'cảm lạnh', 'cảm cúm', 'không được khỏe',
                    'đi khám', 'tái khám', 'xét nghiệm', 'chụp phim', 'siêu âm'
                ],
                'color': '#dc3545',  # Đỏ
                'icon': '🤒',
                'label': 'Đau ốm'
            },
            'special_leave': {
                'keywords': [
                    'thai sản', 'sinh con', 'maternity', 'paternity', 'đám cưới', 'cưới',
                    'wedding', 'đám tang', 'tang lễ', 'funeral', 'ma chay', 'hiếu hỷ',
                    'gia đình', 'bố', 'mẹ', 'con', 'vợ', 'chồng', 'ông', 'bà', 'cháu',
                    'họp mặt gia đình', 'việc gia đình', 'chăm sóc', 'người thân',
                    'khẩn cấp', 'gấp', 'emergency', 'cứu cấp', 'tai nạn', 'sự cố',
                    'bất ngờ', 'đột xuất'
                ],
                'color': '#e83e8c',  # Hồng
                'icon': '👨‍👩‍👧‍👦',
                'label': 'Chế độ đặc biệt'
            }
        }
        
        # Tạo corpus từ keywords
        self.corpus = []
        self.category_names = []
        
        for category, data in self.categories.items():
            combined_text = ' '.join(data['keywords'])
            self.corpus.append(combined_text)
            self.category_names.append(category)
        
        # Khởi tạo TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=None,
            lowercase=True,
            max_features=1000
        )
        
        # Fit vectorizer với corpus
        self.category_vectors = self.vectorizer.fit_transform(self.corpus)
    
    def preprocess_text(self, text: str) -> str:
        """Tiền xử lý text"""
        if not text or pd.isna(text):
            return ""
        
        # Chuyển về lowercase
        text = str(text).lower()
        
        # Loại bỏ dấu câu và ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def classify_reason(self, reason: str, threshold: float = 0.15) -> Dict:
        """
        Phân loại lý do nghỉ bằng cosine similarity với rule-based fallback
        
        Args:
            reason: Lý do nghỉ
            threshold: Ngưỡng similarity tối thiểu
            
        Returns:
            Dict chứa thông tin category
        """
        if not reason or pd.isna(reason):
            return self.get_default_category()
        
        # Tiền xử lý text
        processed_reason = self.preprocess_text(reason)
        
        if not processed_reason:
            return self.get_default_category()
        
        # Rule-based classification trước (cho các trường hợp rõ ràng)
        rule_based_result = self._rule_based_classify(processed_reason)
        if rule_based_result:
            return rule_based_result
        
        try:
            # Vector hóa reason
            reason_vector = self.vectorizer.transform([processed_reason])
            
            # Tính cosine similarity với tất cả categories
            similarities = cosine_similarity(reason_vector, self.category_vectors)[0]
            
            # Tìm category có similarity cao nhất
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            # Kiểm tra threshold
            if max_similarity >= threshold:
                best_category = self.category_names[max_similarity_idx]
                category_info = self.categories[best_category].copy()
                category_info['similarity'] = max_similarity
                category_info['category'] = best_category
                return category_info
            else:
                return self.get_default_category()
                
        except Exception as e:
            print(f"Error in classify_reason: {e}")
            return self.get_default_category()
    
    def _rule_based_classify(self, processed_reason: str) -> Optional[Dict]:
        """
        Rule-based classification cho các trường hợp rõ ràng
        
        Args:
            processed_reason: Lý do đã được tiền xử lý
            
        Returns:
            Dict chứa thông tin category nếu match, None nếu không
        """
        # Sick leave patterns (ưu tiên cao nhất)
        sick_patterns = [
            r'\b(ốm|bệnh|đau|sốt|ho|cảm|không khỏe|sick|ill|fever)\b',
            r'\b(khám bệnh|chữa bệnh|bác sĩ|bệnh viện|phòng khám|doctor|hospital)\b',
            r'\b(thuốc|điều trị|y tế|sức khỏe|medical)\b'
        ]
        
        for pattern in sick_patterns:
            if re.search(pattern, processed_reason, re.IGNORECASE):
                sick_info = self.categories['sick'].copy()
                sick_info['similarity'] = 0.95  # High confidence for rule-based
                sick_info['category'] = 'sick'
                return sick_info
        
        # Remote work patterns
        remote_patterns = [
            r'\b(remote|wfh|work from home|làm việc tại nhà|làm việc từ xa)\b',
            r'\b(ở nhà làm việc|không đến công ty|home office)\b'
        ]
        
        for pattern in remote_patterns:
            if re.search(pattern, processed_reason, re.IGNORECASE):
                remote_info = self.categories['remote'].copy()
                remote_info['similarity'] = 0.90
                remote_info['category'] = 'remote'
                return remote_info
        
        # Business trip patterns
        business_patterns = [
            r'\b(công tác|business trip|meeting|họp|hội nghị)\b',
            r'\b(gặp khách hàng|partner|đối tác|conference)\b',
            r'\b(ra ngoài công tác|đi công tác)\b'
        ]
        
        for pattern in business_patterns:
            if re.search(pattern, processed_reason, re.IGNORECASE):
                business_info = self.categories['business'].copy()
                business_info['similarity'] = 0.88
                business_info['category'] = 'business'
                return business_info
        
        return None
    
    def get_default_category(self) -> Dict:
        """Trả về category mặc định"""
        return {
            'color': '#6c757d',  # Xám
            'icon': '📝',
            'label': 'Khác',
            'category': 'other',
            'similarity': 0.0
        }
    
    def get_category_distribution(self, reasons: List[str]) -> Dict:
        """Phân tích phân bố categories từ danh sách reasons"""
        distribution = {}
        
        for reason in reasons:
            if pd.notna(reason) and str(reason).strip():
                result = self.classify_reason(str(reason))
                category = result['category']
                
                if category not in distribution:
                    distribution[category] = {
                        'count': 0,
                        'reasons': [],
                        'color': result['color'],
                        'label': result['label'],
                        'icon': result['icon']
                    }
                
                distribution[category]['count'] += 1
                distribution[category]['reasons'].append(reason)
        
        return distribution

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
    
    def extract_shift_values(self, shifts_data):
        """Extract shift values từ shifts data và trả về list"""
        shift_values = []
        
        if not shifts_data or not isinstance(shifts_data, list):
            return shift_values
        
        for shift_day in shifts_data:
            shifts = shift_day.get('shifts', [])
            for shift in shifts:
                if shift.get('value'):
                    shift_values.append(shift['value'])
        
        return shift_values
    
    def clean_vietnamese_text(self, text):
        """Clean Vietnamese text for column names"""
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        text = text.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
        return text
    
    def convert_timestamp_to_date(self, timestamp):
        """Chuyển timestamp thành datetime với timezone chính xác"""
        if timestamp and timestamp != '0':
            try:
                # Sử dụng timezone Asia/Ho_Chi_Minh để tránh lệch múi giờ
                utc_dt = datetime.fromtimestamp(int(timestamp), tz=pytz.UTC)
                vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
                return utc_dt.astimezone(vietnam_tz)
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
        
        # Special case: nếu metatype là business và ly_do vẫn rỗng, đặt ly_do = "business"
        business_mask = (
            (df_copy['ly_do'] == '') & 
            (df_copy['metatype'] == 'business')
        )
        df_copy.loc[business_mask, 'ly_do'] = 'business'

        # Special case: nếu metatype là outside và ly_do vẫn rỗng, đặt ly_do = "outside"
        outside_mask = (
            (df_copy['ly_do'] == '') & 
            (df_copy['metatype'] == 'outside')
        )
        df_copy.loc[outside_mask, 'ly_do'] = 'remote'
        
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
                
                # Chuyển đổi timestamp thành datetime với timezone chính xác
                start_date = self.convert_timestamp_to_date(timeoff.get('start_date'))
                end_date = self.convert_timestamp_to_date(timeoff.get('end_date'))
                
                # Extract buoi_nghi từ shifts data
                buoi_nghi = self.extract_shift_values(timeoff.get('shifts', []))
                
                timeoff_record = {
                    'id': timeoff.get('id'),
                    'employee_name': employee_name,
                    'username': username,
                    'state': timeoff.get('state'),
                    'metatype': timeoff.get('metatype'),
                    'paid_timeoff': timeoff.get('paid_timeoff'),
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_leave_days': total_leave_days,
                    'total_shifts': total_shifts,
                    'buoi_nghi': buoi_nghi,
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
    
    def get_shift_time_range(self, buoi_nghi_list):
        """Phân tích buổi nghỉ và trả về thông tin thời gian - UPGRADED VERSION"""
        if not buoi_nghi_list or not isinstance(buoi_nghi_list, list):
            return {'is_all_day': True, 'start_time': None, 'end_time': None}

        if len(buoi_nghi_list) >= 2:
            # Nếu nghỉ cả ngày (cả 2 buổi), trả về danh sách 2 sự kiện
            return {
                'is_all_day': True,
                'shift_events': [
                    {'start_time': '08:00:00', 'end_time': '12:00:00'},
                    {'start_time': '13:00:00', 'end_time': '17:30:00'}
                ]
            }

        if len(buoi_nghi_list) == 1:
            shift = buoi_nghi_list[0]
            shift_time_mapping = {
                '8:00-12:00': {'start_time': '08:00:00', 'end_time': '12:00:00'},
                '13:00-17:30': {'start_time': '13:00:00', 'end_time': '17:30:00'}
            }

            if shift in shift_time_mapping:
                time_info = shift_time_mapping[shift]
                return {
                    'is_all_day': False,
                    'start_time': time_info['start_time'],
                    'end_time': time_info['end_time']
                }

        return {'is_all_day': True, 'start_time': None, 'end_time': None}
    
    def process_and_structure_timeoff(self, row: pd.Series, classifier: ReasonClassifier) -> Optional[List[Dict]]:
        """
        Xử lý chi tiết một yêu cầu nghỉ và trả về một list các bản ghi đã được cấu trúc
        UPGRADED: Xử lý nghỉ nhiều ngày và tạo nhiều events cho mỗi buổi
        """
        if pd.isna(row['start_date']) or pd.isna(row['end_date']):
            return None

        # Phân loại lý do
        reason_result = classifier.classify_reason(str(row['ly_do'])) if row['ly_do'] and str(row['ly_do']).strip() else classifier.get_default_category()

        # Tạo tiêu đề cơ bản
        base_title = f"{reason_result['icon']} {row['employee_name']}"
        if row['ly_do'] and row['ly_do'] != '':
            reason_short = row['ly_do'][:50] + "..." if len(row['ly_do']) > 50 else row['ly_do']
            base_title += f" - {reason_short}"
        base_title += f" ({reason_result['label']})"

        # Xử lý thời gian
        buoi_nghi = row.get('buoi_nghi', [])
        time_info = self.get_shift_time_range(buoi_nghi)

        start_date = row['start_date'].date()
        end_date = row['end_date'].date()
        num_days = (end_date - start_date).days + 1

        processed_leaves = []

        # Tạo events cho từng ngày
        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            day_label = f"Ngày {day_offset + 1}/{num_days}" if num_days > 1 else ""
            
            if time_info.get('shift_events'):
                # Nghỉ cả ngày - tạo 2 bản ghi cho sáng và chiều
                for i, shift_time in enumerate(time_info['shift_events']):
                    shift_label = "Sáng" if i == 0 else "Chiều"
                    title_with_day = f"{base_title}"
                    if day_label:
                        title_with_day += f" - {day_label} - {shift_label}"
                    
                    processed_leaves.append({
                        'title': title_with_day,
                        'start': f"{current_date.strftime('%Y-%m-%d')}T{shift_time['start_time']}",
                        'end': f"{current_date.strftime('%Y-%m-%d')}T{shift_time['end_time']}",
                        'color': reason_result['color'],
                        'borderColor': reason_result['color'],
                        'textColor': '#ffffff',
                        'allDay': False,
                        'extendedProps': {
                            'id': row['id'],
                            'employee': row['employee_name'],
                            'state': row['state'],
                            'metatype': row['metatype'],
                            'days': row['total_leave_days'],
                            'reason': row['ly_do'],
                            'buoi_nghi': buoi_nghi,
                            'approver': row['final_approver'],
                            'created_time': row['created_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['created_time']) else 'N/A',
                            'last_update': row['last_update'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['last_update']) else 'N/A',
                            'paid': row['paid_timeoff'] if 'paid_timeoff' in row else False,
                            'classification': reason_result['label'],
                            'similarity_score': reason_result.get('similarity', 0),
                            'day_info': day_label,
                            'shift_info': shift_label
                        },
                        'display': 'block'
                    })
            elif time_info['is_all_day']:
                # Nghỉ cả ngày - tạo 1 bản ghi all-day
                title_with_day = f"{base_title}"
                if day_label:
                    title_with_day += f" - {day_label}"
                    
                processed_leaves.append({
                    'title': title_with_day,
                    'start': current_date.strftime('%Y-%m-%d'),
                    'end': (current_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'color': reason_result['color'],
                    'borderColor': reason_result['color'],
                    'textColor': '#ffffff',
                    'allDay': True,
                    'extendedProps': {
                        'id': row['id'],
                        'employee': row['employee_name'],
                        'state': row['state'],
                        'metatype': row['metatype'],
                        'days': row['total_leave_days'],
                        'reason': row['ly_do'],
                        'buoi_nghi': buoi_nghi,
                        'approver': row['final_approver'],
                        'created_time': row['created_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['created_time']) else 'N/A',
                        'last_update': row['last_update'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['last_update']) else 'N/A',
                        'paid': row['paid_timeoff'] if 'paid_timeoff' in row else False,
                        'classification': reason_result['label'],
                        'similarity_score': reason_result.get('similarity', 0),
                        'day_info': day_label
                    },
                    'display': 'block'
                })
            else:
                # Nghỉ một buổi cụ thể
                title_with_day = f"{base_title}"
                if day_label:
                    title_with_day += f" - {day_label}"
                    
                processed_leaves.append({
                    'title': title_with_day,
                    'start': f"{current_date.strftime('%Y-%m-%d')}T{time_info['start_time']}",
                    'end': f"{current_date.strftime('%Y-%m-%d')}T{time_info['end_time']}",
                    'color': reason_result['color'],
                    'borderColor': reason_result['color'],
                    'textColor': '#ffffff',
                    'allDay': False,
                    'extendedProps': {
                        'id': row['id'],
                        'employee': row['employee_name'],
                        'state': row['state'],
                        'metatype': row['metatype'],
                        'days': row['total_leave_days'],
                        'reason': row['ly_do'],
                        'buoi_nghi': buoi_nghi,
                        'approver': row['final_approver'],
                        'created_time': row['created_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['created_time']) else 'N/A',
                        'last_update': row['last_update'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['last_update']) else 'N/A',
                        'paid': row['paid_timeoff'] if 'paid_timeoff' in row else False,
                        'classification': reason_result['label'],
                        'similarity_score': reason_result.get('similarity', 0),
                        'day_info': day_label
                    },
                    'display': 'block'
                })

        return processed_leaves

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
        return df, processor
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return pd.DataFrame(), None

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

def convert_df_to_calendar_events(df, processor, use_reason_classification=True):
    """
    UPGRADED VERSION: Chuyển DataFrame thành format events cho calendar 
    với xử lý nghỉ nhiều ngày và tạo nhiều events cho mỗi buổi
    """
    events = []
    
    if df.empty or processor is None:
        return events
    
    # Khởi tạo classifier
    classifier = ReasonClassifier() if use_reason_classification else None
    
    if not use_reason_classification:
        # Fallback về logic cũ nếu không dùng AI
        state_info = get_state_info()
        metatype_info = get_metatype_info()
        
        for _, row in df.iterrows():
            if pd.notna(row['start_date']) and pd.notna(row['end_date']):
                # Logic cũ...
                if row['state'] == 'approved':
                    color = metatype_info.get(row['metatype'], {}).get('color', '#28a745')
                    icon = metatype_info.get(row['metatype'], {}).get('icon', '📅')
                else:
                    color = state_info.get(row['state'], {}).get('color', '#007bff')
                    icon = state_info.get(row['state'], {}).get('icon', '📅')
                
                title = f"{icon} {row['employee_name']}"
                if row['ly_do'] and row['ly_do'] != '':
                    reason_short = row['ly_do'][:25] + "..." if len(row['ly_do']) > 25 else row['ly_do']
                    title += f" - {reason_short}"
                
                events.append({
                    "title": title,
                    "start": row['start_date'].strftime('%Y-%m-%d'),
                    "end": (row['end_date'] + timedelta(days=1)).strftime('%Y-%m-%d'),
                    "color": color,
                    "allDay": True
                })
    else:
        # Sử dụng logic mới với AI classification
        for _, row in df.iterrows():
            processed_events = processor.process_and_structure_timeoff(row, classifier)
            if processed_events:
                events.extend(processed_events)
    
    return events

def display_calendar_legend(show_reason_classification=True):
    """Hiển thị chú thích màu sắc cho calendar"""
    st.markdown("#### 📋 Chú thích")
    
    if show_reason_classification:
        st.markdown("**🎯 Phân loại theo lý do (AI Classification):**")
        
        reason_classifier = ReasonClassifier()
        
        col1, col2 = st.columns(2)
        
        categories = list(reason_classifier.categories.items())
        categories.append(('other', {
            'color': '#6c757d',
            'icon': '📝', 
            'label': 'Khác'
        }))
        
        mid_point = len(categories) // 2 + 1
        
        with col1:
            for category, info in categories[:mid_point]:
                st.markdown(f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                           f"<div style='width: 20px; height: 20px; background-color: {info['color']}; "
                           f"border-radius: 3px; margin-right: 10px;'></div>"
                           f"<span>{info['icon']} {info['label']}</span></div>", 
                           unsafe_allow_html=True)
        
        with col2:
            for category, info in categories[mid_point:]:
                st.markdown(f"<div style='display: flex; align-items: center; margin: 5px 0;'>"
                           f"<div style='width: 20px; height: 20px; background-color: {info['color']}; "
                           f"border-radius: 3px; margin-right: 10px;'></div>"
                           f"<span>{info['icon']} {info['label']}</span></div>", 
                           unsafe_allow_html=True)
    else:
        st.info("💡 Bật 'Sử dụng AI phân loại lý do' để xem chú thích màu sắc thông minh")

def display_event_details(event_data):
    """Hiển thị chi tiết event khi click - UPGRADED với thông tin ngày và buổi"""
    if not event_data:
        return
    
    props = event_data.get('extendedProps', {})
    state_info = get_state_info()
    metatype_info = get_metatype_info()
    
    state = props.get('state', '')
    metatype = props.get('metatype', '')
    state_display = state_info.get(state, {})
    metatype_display = metatype_info.get(metatype, {})
    
    st.markdown("### 📋 Chi tiết yêu cầu nghỉ phép")
    
    # Header card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; margin: 10px 0;">
        <h4 style="margin: 0; color: white;">👤 {props.get('employee', 'N/A')}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📅 Thông tin thời gian:**")
        
        is_all_day = event_data.get('allDay', True)
        start_time = event_data.get('start', 'N/A')
        end_time = event_data.get('end', 'N/A')
        
        # Hiển thị thông tin ngày và buổi nếu có
        day_info = props.get('day_info', '')
        shift_info = props.get('shift_info', '')
        
        if day_info:
            st.info(f"**📆 {day_info}**")
        if shift_info:
            st.success(f"**⏰ Buổi: {shift_info}**")
        
        if is_all_day:
            st.info(f"**Từ ngày:** {start_time}\n"
                    f"**Đến ngày:** {end_time}\n"
                    f"**Số ngày:** {props.get('days', 0)} ngày\n"
                    f"**Loại:** Cả ngày")
        else:
            st.info(f"**Ngày:** {start_time.split('T')[0] if 'T' in start_time else start_time}\n"
                    f"**Thời gian:** {start_time.split('T')[1][:5] if 'T' in start_time else 'N/A'} - {end_time.split('T')[1][:5] if 'T' in end_time else 'N/A'}\n"
                    f"**Số ngày:** {props.get('days', 0)} ngày\n"
                    f"**Loại:** Theo giờ")
        
        buoi_nghi = props.get('buoi_nghi', [])
        if buoi_nghi and isinstance(buoi_nghi, list):
            buoi_nghi_str = ', '.join(buoi_nghi)
            st.success(f"**⏰ Buổi nghỉ:** {buoi_nghi_str}")
        
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
            
            classification = props.get('classification', '')
            similarity_score = props.get('similarity_score', 0)
            if classification and similarity_score > 0:
                st.info(f"**AI Classification:** {classification}")
                st.text(f"Độ chính xác: {similarity_score:.2f}")
        else:
            st.info("**Lý do:** Không có thông tin")
        
        approver = props.get('approver', 'N/A')
        if approver and approver.strip():
            st.success(f"**Người duyệt:** {approver}")
        
        paid_status = "Có lương" if props.get('paid', False) else "Không lương"
        st.info(f"**Loại:** {paid_status}")
    
    st.markdown("**⏰ Thời gian xử lý:**")
    col3, col4 = st.columns(2)
    with col3:
        st.text(f"Tạo: {props.get('created_time', 'N/A')}")
    with col4:
        st.text(f"Cập nhật: {props.get('last_update', 'N/A')}")

def display_reason_analysis(df):
    """Hiển thị phân tích lý do nghỉ phép"""
    st.markdown("### 🤖 Phân tích lý do nghỉ phép (AI Analysis)")
    
    if df.empty or 'ly_do' not in df.columns:
        st.info("Không có dữ liệu lý do để phân tích")
        return
    
    df_with_reason = df[df['ly_do'].notna() & (df['ly_do'].astype(str).str.strip() != '')]
    
    if df_with_reason.empty:
        st.info("Không có lý do nghỉ phép trong dữ liệu")
        return
    
    classifier = ReasonClassifier()
    reasons_list = df_with_reason['ly_do'].astype(str).tolist()
    distribution = classifier.get_category_distribution(reasons_list)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if distribution:
            categories = []
            counts = []
            colors = []
            
            for category, data in distribution.items():
                categories.append(f"{data['icon']} {data['label']}")
                counts.append(data['count'])
                colors.append(data['color'])
            
            fig = px.pie(
                values=counts,
                names=categories,
                title="🎯 Phân bố lý do nghỉ phép (AI Classification)",
                color_discrete_sequence=colors
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**📊 Thống kê chi tiết:**")
        
        total_with_reason = len(df_with_reason)
        total_classified = sum(data['count'] for data in distribution.values())
        
        st.metric("Tổng số có lý do", total_with_reason)
        st.metric("Đã phân loại", total_classified)
        st.metric("Tỷ lệ phân loại", f"{total_classified/total_with_reason*100:.1f}%")
        
        sorted_categories = sorted(distribution.items(), key=lambda x: x[1]['count'], reverse=True)
        
        st.markdown("**🏆 Top categories:**")
        for i, (category, data) in enumerate(sorted_categories[:5]):
            percentage = (data['count'] / total_classified * 100) if total_classified > 0 else 0
            st.markdown(f"**{i+1}.** {data['icon']} {data['label']}: {data['count']} ({percentage:.1f}%)")

def display_buoi_nghi_analysis(df):
    """Hiển thị phân tích buổi nghỉ"""
    st.markdown("### ⏰ Phân tích buổi nghỉ")
    
    if df.empty or 'buoi_nghi' not in df.columns:
        st.info("Không có dữ liệu buổi nghỉ để phân tích")
        return
    
    df_with_buoi = df[df['buoi_nghi'].notna() & (df['buoi_nghi'].astype(str) != '[]')]
    
    if df_with_buoi.empty:
        st.info("Không có dữ liệu buổi nghỉ")
        return
    
    shift_counts = {}
    shift_combinations = {}
    
    for idx, row in df_with_buoi.iterrows():
        buoi_nghi = row['buoi_nghi']
        if isinstance(buoi_nghi, list) and buoi_nghi:
            for shift in buoi_nghi:
                if shift not in shift_counts:
                    shift_counts[shift] = 0
                shift_counts[shift] += 1
            
            combination_key = ' + '.join(sorted(buoi_nghi))
            if combination_key not in shift_combinations:
                shift_combinations[combination_key] = 0
            shift_combinations[combination_key] += 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        if shift_counts:
            shifts = list(shift_counts.keys())
            counts = list(shift_counts.values())
            
            fig1 = px.bar(
                x=shifts,
                y=counts,
                title="📊 Tần suất buổi nghỉ",
                labels={'x': 'Buổi', 'y': 'Số lần'},
                color=counts,
                color_continuous_scale="Blues"
            )
            fig1.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if shift_combinations:
            combinations = list(shift_combinations.keys())[:10]
            combo_counts = [shift_combinations[combo] for combo in combinations]
            
            fig2 = px.bar(
                x=combo_counts,
                y=combinations,
                orientation='h',
                title="🔄 Top 10 kết hợp buổi nghỉ",
                labels={'x': 'Số lần', 'y': 'Kết hợp'},
                color=combo_counts,
                color_continuous_scale="Viridis"
            )
            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### 📈 Thống kê buổi nghỉ:")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        total_shifts = sum(shift_counts.values())
        st.metric("Tổng số buổi nghỉ", total_shifts)
    
    with col4:
        unique_shifts = len(shift_counts)
        st.metric("Số loại buổi khác nhau", unique_shifts)
    
    with col5:
        avg_shifts_per_request = total_shifts / len(df_with_buoi)
        st.metric("Trung bình buổi/yêu cầu", f"{avg_shifts_per_request:.1f}")

def main():
    """Main dashboard - UPGRADED VERSION"""
    
    # Custom CSS
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
    
    .upgrade-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏖️ Time Off Management Dashboard <span class="upgrade-badge">✨ UPGRADED</span></h1>
        <p>Quản lý và theo dõi yêu cầu nghỉ phép với AI Classification & Multi-day Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Đang tải dữ liệu..."):
        df, processor = load_timeoff_data()
    
    if df.empty:
        st.error("❌ Không thể tải dữ liệu timeoff")
        return
    
    # Sidebar filters
    st.sidebar.markdown("## 🔍 Bộ lọc dữ liệu")
    
    use_ai_classification = st.sidebar.checkbox("🤖 Sử dụng AI phân loại lý do", value=True)
    
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
    
    # State filter
    state_info = get_state_info()
    state_options = ['Tất cả'] + [f"{info['icon']} {info['label']}" for state, info in state_info.items()]
    selected_states_display = st.sidebar.multiselect(
        "📊 Trạng thái",
        state_options,
        default=['Tất cả']
    )
    
    selected_states = []
    if 'Tất cả' not in selected_states_display:
        for display in selected_states_display:
            for state, info in state_info.items():
                if f"{info['icon']} {info['label']}" == display:
                    selected_states.append(state)
    else:
        selected_states = ['Tất cả']
    
    # Metatype filter
    metatype_info = get_metatype_info()
    metatype_options = ['Tất cả'] + [f"{info['icon']} {info['label']}" for meta, info in metatype_info.items()]
    selected_metatypes_display = st.sidebar.multiselect(
        "📋 Loại nghỉ phép",
        metatype_options,
        default=['Tất cả']
    )
    
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
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['start_date'].dt.date >= date_range[0]) & 
            (filtered_df['start_date'].dt.date <= date_range[1])
        ]
    
    if 'Tất cả' not in selected_employees:
        filtered_df = filtered_df[filtered_df['employee_name'].isin(selected_employees)]
    
    if 'Tất cả' not in selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
    
    if 'Tất cả' not in selected_metatypes:
        filtered_df = filtered_df[filtered_df['metatype'].isin(selected_metatypes)]
    
    # Summary metrics
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
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Calendar View", 
        "📊 Analytics", 
        "🤖 AI Analysis",
        "⏰ Shift Analysis",
        "📋 Data Table", 
        "⚙️ Settings"
    ])
    
    with tab1:
        st.markdown("### 📅 Lịch Time Off")
        st.info("✨ UPGRADED: Hỗ trợ hiển thị nghỉ nhiều ngày, tạo nhiều events cho mỗi buổi nghỉ")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            mode_options = {
                "dayGridMonth": "📅 Tháng",
                "dayGridWeek": "📅 Tuần", 
                "timeGridWeek": "⏰ Tuần (giờ)",
                "listMonth": "📋 Danh sách tháng"
            }
            
            selected_mode_display = st.selectbox(
                "Chế độ xem:",
                list(mode_options.values()),
                index=2  # Default to timeGridWeek để thấy rõ events theo giờ
            )
            
            mode = [k for k, v in mode_options.items() if v == selected_mode_display][0]
        
        with col2:
            st.markdown("**🎨 Tùy chọn hiển thị**")
            show_legend = st.checkbox("Hiển thị chú thích", value=True)
            show_weekend = st.checkbox("Hiển thị cuối tuần", value=True)
        
        # Convert data to events
        events = convert_df_to_calendar_events(filtered_df, processor, use_reason_classification=use_ai_classification)
        
        # Calendar options
        calendar_options = {
            "editable": False,
            "navLinks": True,
            "selectable": False,
            "dayMaxEvents": 5,
            "moreLinkClick": "popover",
            "eventDisplay": "block",
            "displayEventTime": True,
            "weekends": show_weekend,
            "headerToolbar": {
                "left": "today prev,next",
                "center": "title", 
                "right": "dayGridMonth,dayGridWeek,timeGridWeek,listMonth"
            },
            "initialView": mode,
            "height": 700,
            "locale": "vi",
            "slotMinTime": "06:00:00",
            "slotMaxTime": "20:00:00",
            "slotDuration": "00:30:00",
            "expandRows": True
        }
        
        # Custom CSS
        custom_css = """
        <style>
        .fc-event {
            font-size: 12px;
            border-radius: 4px;
            padding: 2px 4px;
            margin: 1px 0;
        }
        .fc-timegrid-event {
            border-left-width: 3px !important;
        }
        </style>
        """
        
        st.markdown(custom_css, unsafe_allow_html=True)
        
        # Display calendar
        if events:
            st.markdown(f"**📊 Tổng số events hiển thị: {len(events)}**")
            st.markdown('<div class="calendar-container">', unsafe_allow_html=True)
            
            calendar_state = calendar(
                events=events,
                options=calendar_options,
                custom_css=custom_css,
                key="timeoff_calendar_upgraded"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if calendar_state.get("eventClick"):
                event_data = calendar_state["eventClick"]["event"]
                display_event_details(event_data)
                
        else:
            st.info("📅 Không có dữ liệu time off trong khoảng thời gian được chọn")
        
        if show_legend:
            display_calendar_legend(show_reason_classification=use_ai_classification)
    
    with tab2:
        st.subheader("📊 Phân tích dữ liệu")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                state_counts = filtered_df['state'].value_counts()
                colors = [get_state_info().get(state, {}).get('color', '#007bff') for state in state_counts.index]
                
                fig1 = px.pie(
                    values=state_counts.values, 
                    names=[get_state_info().get(state, {}).get('label', state) for state in state_counts.index],
                    title="🎯 Phân bố theo trạng thái",
                    color_discrete_sequence=colors
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
                
                top_employees = filtered_df['employee_name'].value_counts().head(10)
                fig3 = px.bar(
                    x=top_employees.values,
                    y=top_employees.index,
                    orientation='h',
                    title="👥 Top 10 nhân viên",
                    color=top_employees.values,
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                metatype_counts = filtered_df['metatype'].value_counts()
                colors = [get_metatype_info().get(meta, {}).get('color', '#007bff') for meta in metatype_counts.index]
                
                fig2 = px.bar(
                    x=[get_metatype_info().get(meta, {}).get('label', meta) for meta in metatype_counts.index],
                    y=metatype_counts.values,
                    title="📋 Phân bố theo loại",
                    color=metatype_counts.values,
                    color_discrete_sequence=colors
                )
                st.plotly_chart(fig2, use_container_width=True)
                
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
                    st.plotly_chart(fig4, use_container_width=True)
    
    with tab3:
        display_reason_analysis(filtered_df)
    
    with tab4:
        display_buoi_nghi_analysis(filtered_df)
    
    with tab5:
        st.subheader("📋 Bảng dữ liệu")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            show_all_columns = st.checkbox("Hiển thị tất cả cột", False)
        with col3:
            items_per_page = st.selectbox("Số dòng/trang", [10, 25, 50, 100], index=1)
        
        if not filtered_df.empty:
            if show_all_columns:
                display_df = filtered_df
            else:
                important_cols = [
                    'employee_name', 'state', 'metatype', 'start_date', 'end_date', 
                    'total_leave_days', 'buoi_nghi', 'ly_do', 'final_approver'
                ]
                available_cols = [col for col in important_cols if col in filtered_df.columns]
                display_df = filtered_df[available_cols]
            
            if 'start_date' in display_df.columns:
                display_df['start_date'] = pd.to_datetime(display_df['start_date']).dt.strftime('%Y-%m-%d')
            if 'end_date' in display_df.columns:
                display_df['end_date'] = pd.to_datetime(display_df['end_date']).dt.strftime('%Y-%m-%d')
            
            if 'buoi_nghi' in display_df.columns:
                display_df_copy = display_df.copy()
                display_df_copy['buoi_nghi'] = display_df_copy['buoi_nghi'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) and x else 'N/A'
                )
                display_df = display_df_copy
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("### 📥 Tải xuống dữ liệu")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Tải xuống CSV",
                    data=csv,
                    file_name=f"timeoff_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                json_data = filtered_df.to_json(orient='records', date_format='iso', force_ascii=False)
                st.download_button(
                    label="📥 Tải xuống JSON",
                    data=json_data,
                    file_name=f"timeoff_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    
    with tab6:
        st.subheader("⚙️ Cài đặt hệ thống")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Thông tin hệ thống")
            st.metric("Tổng số records", len(df))
            st.metric("Số nhân viên", df['employee_name'].nunique())
            st.metric("Records sau lọc", len(filtered_df))
            
            # Upgrade features
            st.markdown("#### ✨ Tính năng mới")
            st.success("✅ Xử lý timezone chính xác (pytz)")
            st.success("✅ Hỗ trợ nghỉ nhiều ngày")
            st.success("✅ Tạo nhiều events cho mỗi buổi")
            st.success("✅ Hiển thị chi tiết ngày và buổi")
        
        with col2:
            st.markdown("#### 🔧 Công cụ quản lý")
            
            if st.button("🔄 Làm mới dữ liệu", type="primary", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            st.markdown("---")
            
            full_csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải xuống toàn bộ dữ liệu",
                data=full_csv,
                file_name=f"full_timeoff_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("**⚡ Trạng thái:**")
            st.success("✅ API Connection: Active")
            st.success("✅ Data Cache: Active")
            st.success("✅ Timezone: Asia/Ho_Chi_Minh")
            st.success("✅ Multi-day Support: Enabled")
            st.info(f"🕒 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
