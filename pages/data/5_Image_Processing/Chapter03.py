import numpy as np
import cv2

L = 256

def Negative(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imgout[x, y] = np.uint8(s)
    return imgout

def NegativeColor(imgin):
    # C: chanel R G B
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y, 2]
            g = imgin[x, y, 1]
            b = imgin[x, y, 0]
            r = L - 1 - r
            g = L - 1 - g
            b = L - 1 - b
            imgout[x, y, 2] = np.uint8(r)
            imgout[x, y, 1] = np.uint8(g)
            imgout[x, y, 0] = np.uint8(b)
    return imgout

def Logarit(imgin):
    # làm cho ảnh trở nên sáng hơn
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    c = (L - 1) / np.log(1.0 * L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r == 0:
                r = 1
            s = c * np.log(1.0 + r)
            imgout[x, y] = np.uint8(s)
    return imgout

def LogaritColor(imgin):
    # làm cho ảnh trở nên sáng hơn
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), np.uint8) + 255
    c = (L - 1) / np.log(1.0 * L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y, 2]
            g = imgin[x, y, 1]
            b = imgin[x, y, 0]
            if r == 0:
                r = 1
            if g == 0:
                g = 1
            if b == 0:
                b = 1
            r = c * np.log(1.0 + r)
            g = c * np.log(1.0 + g)
            b = c * np.log(1.0 + b)
            imgout[x, y, 2] = np.uint8(r)
            imgout[x, y, 1] = np.uint8(g)
            imgout[x, y, 0] = np.uint8(b)
    return imgout

def Power(imgin):
    # làm cho ảnh trở nên tối đi
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    gamma = 5.0
    # gamma > 1 -> làm tăng độ sáng ảnh
    # gamma < 1 -> làm giảm độ sáng ảnh
    # gamma == 1 -> ảnh giữ nguyên
    # các chỉ số gamma và c được chọn từ đồ thị (xem sách)
    c = np.power(L - 1.0, 1.0 - gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = c * np.power(r, gamma)
            imgout[x, y] = np.uint8(s)
    return imgout

def PowerColor(imgin):
    # làm cho ảnh trở nên tối đi
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), np.uint8)
    gamma = 5.0
    c = np.power(L - 1.0, 1.0 - gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y, 2]
            g = imgin[x, y, 1]
            b = imgin[x, y, 0]
            r = c * np.power(r, gamma)
            g = c * np.power(g, gamma)
            b = c * np.power(b, gamma)
            imgout[x, y, 2] = np.uint8(r)
            imgout[x, y, 1] = np.uint8(g)
            imgout[x, y, 0] = np.uint8(b)
    return imgout

def PiecewiseLinear(imgin):
    # Tuyến tình từng phần là nối từng đoạn thẳng lại với nhau
    # tiền thân của các filter làm đẹp sau này
    # tinh chỉnh độ tương phản của ảnh
    # thường sẽ là tăng độ tương phản (contrast) của ảnh
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            # Đoạn I
            if r < r1:
                s = s1 * r / r1
            # Đoạn II
            elif r < r2:
                s = ((s2 - s1) * (r - r1) / (r2 - r1)) + s1
            # Đoạn III
            else:
                s = ((L - 1 - s2) * (r - r2) / (L - 1 - r2)) + s2
            imgout[x, y] = np.uint8(s)
    return imgout

def PiecewiseLinearColor(imgin):
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), np.uint8) + 255
    for c in range(C):
        rmin, rmax, _, _ = cv2.minMaxLoc(imgin[:, :, c])
        r1 = rmin
        s1 = 0
        r2 = rmax
        s2 = L - 1
        for x in range(M):
            for y in range(N):
                r = imgin[x, y, c]
                if r < r1:
                    s = s1 * r / r1
                elif r < r2:
                    s = ((s2 - s1) * (r - r1) / (r2 - r1)) + s1
                else:
                    s = ((L - 1 - s2) * (r - r2) / (L - 1 - r2)) + s2
                imgout[x, y, c] = np.uint8(s)
    return imgout

def Histogram(imgin):
    # 3.3
    # Histogram của một ảnh xám có độ sáng nằm trong phạm vi từ [0, L - 1]
    M, N = imgin.shape
    imgout = np.zeros((M, L), np.uint8) + 255
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1
    p = h / (M * N)
    scale = 3000
    for r in range(0, L):
        cv2.line(imgout, (r, M - 1), (r, M - 1 - int(scale * p[r])), (0, 0, 0))
    return imgout

def HistEqual(imgin):
    # 3.3.1
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1
    p = h / (M * N)

    s = np.zeros(L, np.float64)
    for k in range(0, L):
        for j in range(0, k + 1):
            s[k] = s[k] + p[j]

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            imgout[x, y] = np.uint8((L - 1) * s[r])
    return imgout

def HistEqualColor(imgin):
    B = imgin[:, :, 0]
    G = imgin[:, :, 1]
    R = imgin[:, :, 2]
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    imgout = np.array([B, G, R])
    imgout = np.transpose(imgout, axes=[1, 2, 0])
    return imgout

def LocalHist(imgin):
    # 3.3.2
    # Quét ảnh
    # Lấy từng vùng nhỏ có kích thước 3x3
    # Cân bằng histogram của từng vùng nhỏ đó
    # Thay phần tử đang xét bằng phần tử ở giữa
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    m = 3
    n = 3
    w = np.zeros((m, n), np.uint8)
    a = m // 2 
    b = n // 2
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a : x + a + 1, y - b : y + b + 1]
            w = cv2.equalizeHist(w)
            imgout[x, y] = w[a, b]
    return imgout

def HistStat(imgin):
    # 3.3.3. Thống kê histogram
    # Xử lý histogram cục bộ tuy thấy thông tin bên trong, nhưng làm sai lệch ảnh
    # => đưa ra phương pháp chọn điểm ảnh cần làm rõ dựa trên 2 đại lượng thống kê
    # trung bình (mean) và độ lệch chuẩn (standard deviation)
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m, n), np.uint8)
    a = m // 2
    b = n // 2
    mG, sigmaG = cv2.meanStdDev(imgin)
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a : x + a + 1, y - b : y + b + 1]
            msxy, sigmasxy = cv2.meanStdDev(w)
            r = imgin[x, y]
            if (k0 * mG <= msxy <= k1 * mG) and (
                k2 * sigmaG <= sigmasxy <= k3 * sigmaG
            ):
                imgout[x, y] = np.uint8(C * r)
            else:
                imgout[x, y] = r
    return imgout

def MyFilter2D(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    m = 11
    n = 11
    w = np.zeros((m, n), np.float32) + 1 / (m * n)
    a = m // 2
    b = n // 2
    for x in range(0, M):
        for y in range(0, M):
            r = 0.0
            for s in range(-a, a + 1):
                for t in range(-b, b + 1):
                    x_new = (x + s) % M
                    y_new = (y + t) % M
                    r = r + w[s + a, t + b] * imgin[x_new, y_new]
            if r < 0.0:
                r = 0.0
            if r > L - 1:
                r = L - 1
            imgout[x, y] = np.uint8(r)
    return imgout

def MySmooth(imgin):
    m = 11
    n = 11
    w = np.zeros((m, n), np.float32) + 1 / (m * n)
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout

def MyMedian(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 5
    n = 5
    a = m // 2
    b = n // 2
    for x in range(a, M - a):
        for y in range(b, N - b):
            w = imgin[x - a : x + a + 1, y - b : y + b + 1]
            w = w.reshape((m * n))
            w = np.sort(w)
            imgout[x, y] = w[m * n // 2]
    return imgout

def Threshold(imgin):
    temp = cv2.blur(imgin, (15, 15))
    retval, imgout = cv2.threshold(temp, 64, 255, cv2.THRESH_BINARY)
    return imgout

def Sharpen(imgin):
    # Đạo hàm cấp 2 của ảnh
    w = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)

    # Hàm cv2.Laplacian chỉ tính đạo hàm cấp 2
    # cho bộ lọc có số -4 chính giữa
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout

def Gradient(imgin):
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Đạo hàm cấp 1 theo hướng x
    mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
    # Đạo hàm cấp 1 theo hướng y
    mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

    # Lưu ý: cv2.Sobel có hướng x nằm ngang
    # ngược lại với sách Digital Image Processing
    gx = cv2.Sobel(imgin, cv2.CV_32FC1, dx=1, dy=0)
    gy = cv2.Sobel(imgin, cv2.CV_32FC1, dx=0, dy=1)

    imgout = abs(gx) + abs(gy)
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout
