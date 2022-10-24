from scipy.signal import find_peaks


y = [1,1,1,1,5,1,1,1,1,6,1,1,1,7,1,1,1]
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

peaks, _ = find_peaks(y, height=0)

print(y(peaks))