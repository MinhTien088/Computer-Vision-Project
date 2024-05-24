import streamlit as st
import cv2 as cv
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Đồ án Cuối kỳ - Xử lý ảnh số", 
    page_icon="🎓", 
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align: center;">
        <a href="https://fit.hcmute.edu.vn/">
            <img src="https://fit.hcmute.edu.vn/Resources/Images/SubDomain/fit/HCMUTE-fit.png" alt="Khoa Công nghệ Thông tin" class="custom-image-class" style="width: 1000px; height: 126px;">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("---")

with st.container():
    st.write(
        """
        <div>
            <h1 style="text-align: center; padding-bottom: 1px;">ĐỒ ÁN HỌC PHẦN</h1>
            <p style="text-align: center; font-size: 20px; padding-top: 1px;">
                --- MÔN HỌC: XỬ LÝ ẢNH SỐ ---<br>
                --- GVHD: ThS. Trần Tiến Đức ---
            </p>
            <span style="text-align: justify; font-size: 18px;">
                Đồ án xử lý ảnh số là một nghiên cứu chuyên sâu về các phương pháp và kỹ thuật sử dụng máy tính để xử lý hình ảnh. Đồ án này tập trung vào việc áp dụng các thuật toán và phương pháp số nhằm phân tích, cải thiện và hiểu rõ nội dung của hình ảnh.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("---")

with st.container():
    st.header("Thành viên nhóm")
    left_column, right_column = st.columns(2)
    with left_column:
        text_column, image_column = st.columns(2)
        with text_column:
            st.markdown(
                """
                - <p style="font-size: 18px; padding-top: 40px">Tăng Huỳnh Minh Tiến</p>
                - <p style="font-size: 18px;"> MSSV: 21133088</p>
                """,
                unsafe_allow_html=True,
            )
        with image_column:
            st.image(os.path.join(curr_dir, 'pages/data/0_Home/21133088.jpg'), width=150)
    with right_column:
        text_column, image_column = st.columns(2)
        with text_column:
            st.markdown(
                """
                - <p style="font-size: 18px; padding-top: 40px">Nguyễn Thành Trung</p>
                - <p style="font-size: 18px;"> MSSV: 21133090</p>
                """,
                unsafe_allow_html=True,
            )
        with image_column:
            st.image(os.path.join(curr_dir, 'pages/data/0_Home/21133090.jpg'), width=150)

st.write("---")

with st.container():
    st.markdown(
        """
        <div>
            <h2>Tài Liệu Tham Khảo</h2>
            <ul>
                <li><a href="https://docs.streamlit.io/library/get-started">https://docs.streamlit.io/library/get-started</a></li>
            </ul>
            <h2>Liên Hệ</h2>
            <ul>
                <li>Tăng Huỳnh Minh Tiến: <a href="mailto:21133088@student.hcmute.edu.vn">21133088@student.hcmute.edu.vn</a></li>
                <li>Nguyễn Thành Trung: <a href="mailto:21133090@student.hcmute.edu.vn">21133090@student.hcmute.edu.vn</a></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
