from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

def predict_player(image_path):
    model = load_model('foodballplayer.h5')
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions[0])
    class_names = ['Bernardo Silva', 'Công Phượng', 'Đặng Văn Lâm', 'Đoàn Văn Hậu', 'Erling Haaland', 'Gareth Bale', 'Jude Bellingham', 'Karim Benzema',
                   'Kevin De Bruyne', 'Kylian Mbappé', 'Luka Modrić', 'Messi', 'Mohamed Salah', 'Neymar', 'Nguyễn Hoàn Đức', 'Nguyễn Văn Quyết', 'Quang Hải',
                   'Roberto Carlos', 'Rodrygo Goes', 'Ronaldinho', 'Ronaldo', 'Tiến Dũng', 'Văn Toàn', 'Vinícius Júnior']
    predicted_player_name = class_names[predicted_class]
    print('Cầu thủ trong hình là: {}'.format(predicted_player_name))
    plt.imshow(img)
    plt.show()
    return predicted_player_name

image_path = '/content/drive/MyDrive/football player train/DoanVanHau/1.png'
A = predict_player(image_path)
# Tóm tắt tiểu sử cầu thủ
if A == 'Bernardo Silva':
  B = """
Bernardo Silva, tiền vệ tấn công người Bồ Đào Nha, sinh ngày 10 tháng 8 năm 1994. Hiện đang thi đấu cho Manchester City và ĐTQG Bồ Đào Nha.
Sự nghiệp:
•	Bắt đầu từ lò đào tạo trẻ Benfica.
•	Gia nhập Monaco năm 2015, thành công rực rỡ với chức vô địch Ligue 1 và vào bán kết Champions League.
•	Chuyển đến Manchester City năm 2017, góp phần vào 4 chức vô địch Premier League, 1 FA Cup và 6 League Cup.
•	Thành viên quan trọng của ĐTQG Bồ Đào Nha, góp công vào chức vô địch UEFA Nations League 2019.
Phong cách:
•	Kỹ thuật cá nhân điêu luyện, chuyền bóng tốt, nhãn quan chiến thuật sắc bén.
•	Cầu thủ tấn công sáng tạo, tạo cơ hội cho đồng đội ghi bàn.
•	Chăm chỉ, tinh thần đồng đội cao.
Thành tích:
•	4 chức vô địch Premier League
•	1 FA Cup
•	6 League Cup
•	1 UEFA Nations League
•	1 Cầu thủ trẻ xuất sắc nhất Ligue 1
Bernardo Silva là một trong những tiền vệ tấn công xuất sắc nhất thế giới hiện nay với tương lai đầy hứa hẹn.
"""
  print("""
Bernardo Silva, tiền vệ tấn công người Bồ Đào Nha, sinh ngày 10 tháng 8 năm 1994. Hiện đang thi đấu cho Manchester City và ĐTQG Bồ Đào Nha.
Sự nghiệp:
•	Bắt đầu từ lò đào tạo trẻ Benfica.
•	Gia nhập Monaco năm 2015, thành công rực rỡ với chức vô địch Ligue 1 và vào bán kết Champions League.
•	Chuyển đến Manchester City năm 2017, góp phần vào 4 chức vô địch Premier League, 1 FA Cup và 6 League Cup.
•	Thành viên quan trọng của ĐTQG Bồ Đào Nha, góp công vào chức vô địch UEFA Nations League 2019.
Phong cách:
•	Kỹ thuật cá nhân điêu luyện, chuyền bóng tốt, nhãn quan chiến thuật sắc bén.
•	Cầu thủ tấn công sáng tạo, tạo cơ hội cho đồng đội ghi bàn.
•	Chăm chỉ, tinh thần đồng đội cao.
Thành tích:
•	4 chức vô địch Premier League
•	1 FA Cup
•	6 League Cup
•	1 UEFA Nations League
•	1 Cầu thủ trẻ xuất sắc nhất Ligue 1
Bernardo Silva là một trong những tiền vệ tấn công xuất sắc nhất thế giới hiện nay với tương lai đầy hứa hẹn.
""")

if A == 'Công Phượng':
  B = """
Nguyễn Công Phượng (sinh ngày 21 tháng 1 năm 1995) là cầu thủ bóng đá chuyên nghiệp người Việt Nam thi đấu ở vị trí tiền đạo cho câu lạc bộ Hoàng Anh Gia Lai tại V.League 1 và Đội tuyển quốc gia Việt Nam.
Sự nghiệp:
•	Khởi nghiệp tại lò đào tạo trẻ PVF và HAGL.
•	Nổi tiếng với biệt danh "Công Phượng - Messi Việt Nam" khi còn trẻ.
•	Chuyển sang thi đấu cho nhiều câu lạc bộ như: HAGL, Júbilo Iwata (Nhật Bản), Incheon United (Hàn Quốc), TP. Hồ Chí Minh.
•	Quay trở lại HAGL thi đấu từ năm 2020.
•	Thành viên đội tuyển U19 Việt Nam tham dự giải U19 châu Á 2016 và U20 World Cup 2017.
•	Góp công vào thành công của ĐTQG Việt Nam: vô địch AFF Cup 2018, hạng nhì King's Cup 2019.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân tốt, tốc độ nhanh, rê bóng lắt léo.
•	Khả năng dứt điểm đa dạng, thường xuyên ghi bàn đẹp mắt.
•	Có tinh thần thi đấu máu lửa, luôn nỗ lực hết mình.
Thành tích:
•	1 AFF Cup
•	1 HCV SEA Games
•	1 Cầu thủ trẻ xuất sắc nhất Đông Nam Á 2016
•	2 lần Quả bóng vàng Việt Nam (2015, 2016)
Nguyễn Công Phượng là một trong những cầu thủ tài năng và được yêu thích nhất Việt Nam. Anh được kỳ vọng sẽ là niềm tự hào của bóng đá Việt Nam trong tương lai.
"""
  print("""
Nguyễn Công Phượng (sinh ngày 21 tháng 1 năm 1995) là cầu thủ bóng đá chuyên nghiệp người Việt Nam thi đấu ở vị trí tiền đạo cho câu lạc bộ Hoàng Anh Gia Lai tại V.League 1 và Đội tuyển quốc gia Việt Nam.
Sự nghiệp:
•	Khởi nghiệp tại lò đào tạo trẻ PVF và HAGL.
•	Nổi tiếng với biệt danh "Công Phượng - Messi Việt Nam" khi còn trẻ.
•	Chuyển sang thi đấu cho nhiều câu lạc bộ như: HAGL, Júbilo Iwata (Nhật Bản), Incheon United (Hàn Quốc), TP. Hồ Chí Minh.
•	Quay trở lại HAGL thi đấu từ năm 2020.
•	Thành viên đội tuyển U19 Việt Nam tham dự giải U19 châu Á 2016 và U20 World Cup 2017.
•	Góp công vào thành công của ĐTQG Việt Nam: vô địch AFF Cup 2018, hạng nhì King's Cup 2019.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân tốt, tốc độ nhanh, rê bóng lắt léo.
•	Khả năng dứt điểm đa dạng, thường xuyên ghi bàn đẹp mắt.
•	Có tinh thần thi đấu máu lửa, luôn nỗ lực hết mình.
Thành tích:
•	1 AFF Cup
•	1 HCV SEA Games
•	1 Cầu thủ trẻ xuất sắc nhất Đông Nam Á 2016
•	2 lần Quả bóng vàng Việt Nam (2015, 2016)
Nguyễn Công Phượng là một trong những cầu thủ tài năng và được yêu thích nhất Việt Nam. Anh được kỳ vọng sẽ là niềm tự hào của bóng đá Việt Nam trong tương lai.
""")
if A == 'Đặng Văn Lâm':
  B = """
Đặng Văn Lâm (sinh ngày 13 tháng 8 năm 1993) là cầu thủ bóng đá chuyên nghiệp người Việt Nam gốc Nga đang thi đấu ở vị trí thủ môn cho câu lạc bộ Cerezo Osaka tại J1 League và Đội tuyển quốc gia Việt Nam.
Sự nghiệp:
•	Khởi nghiệp tại Spartak Moscow và Dinamo Moscow (Nga).
•	Chuyển đến Việt Nam thi đấu cho Hoàng Anh Gia Lai và Hải Phòng.
•	Nổi tiếng với biệt danh "Lâm Tây" và là thủ môn số 1 của ĐTQG Việt Nam.
•	Thi đấu ấn tượng tại AFF Cup 2018, Asian Cup 2019 và Vòng loại World Cup 2022.
•	Chuyển sang Cerezo Osaka vào năm 2020.
Phong cách chơi bóng:
•	Phản xạ nhanh nhẹn, đổ người tốt, bắt bóng chắc chắn.
•	Khả năng ra vào hợp lý, chỉ huy tốt hàng thủ.
•	Có tinh thần thi đấu quả cảm, không ngại va chạm.
Thành tích:
•	1 AFF Cup
•	1 HCV SEA Games
•	1 Cầu thủ xuất sắc nhất AFF Cup 2018
Đặng Văn Lâm là một trong những thủ môn xuất sắc nhất Việt Nam hiện nay và được kỳ vọng sẽ tiếp tục cống hiến cho ĐTQG trong tương lai.
"""
  print("""
Đặng Văn Lâm (sinh ngày 13 tháng 8 năm 1993) là cầu thủ bóng đá chuyên nghiệp người Việt Nam gốc Nga đang thi đấu ở vị trí thủ môn cho câu lạc bộ Cerezo Osaka tại J1 League và Đội tuyển quốc gia Việt Nam.
Sự nghiệp:
•	Khởi nghiệp tại Spartak Moscow và Dinamo Moscow (Nga).
•	Chuyển đến Việt Nam thi đấu cho Hoàng Anh Gia Lai và Hải Phòng.
•	Nổi tiếng với biệt danh "Lâm Tây" và là thủ môn số 1 của ĐTQG Việt Nam.
•	Thi đấu ấn tượng tại AFF Cup 2018, Asian Cup 2019 và Vòng loại World Cup 2022.
•	Chuyển sang Cerezo Osaka vào năm 2020.
Phong cách chơi bóng:
•	Phản xạ nhanh nhẹn, đổ người tốt, bắt bóng chắc chắn.
•	Khả năng ra vào hợp lý, chỉ huy tốt hàng thủ.
•	Có tinh thần thi đấu quả cảm, không ngại va chạm.
Thành tích:
•	1 AFF Cup
•	1 HCV SEA Games
•	1 Cầu thủ xuất sắc nhất AFF Cup 2018
Đặng Văn Lâm là một trong những thủ môn xuất sắc nhất Việt Nam hiện nay và được kỳ vọng sẽ tiếp tục cống hiến cho ĐTQG trong tương lai.
""")
if A == 'Đoàn Văn Hậu':
  B ="""
Đoàn Văn Hậu (sinh ngày 19 tháng 4 năm 1999) là cầu thủ bóng đá chuyên nghiệp người Việt Nam chơi ở vị trí hậu vệ trái cho câu lạc bộ Hà Nội và Đội tuyển quốc gia Việt Nam.
Sự nghiệp:
•	Khởi nghiệp tại lò đào tạo trẻ Hà Nội T&T.
•	Nổi tiếng với biệt danh "Hậu tủ" và là một trong những hậu vệ trái xuất sắc nhất Việt Nam.
•	Thi đấu ấn tượng tại V.League 1, AFC Champions League và các giải đấu quốc tế cùng ĐTQG Việt Nam.
•	Chuyển sang SC Heerenveen (Hà Lan) vào năm 2019 nhưng gặp chấn thương và trở về Việt Nam thi đấu cho Hà Nội FC.
•	Góp công vào thành công của ĐTQG Việt Nam: vô địch AFF Cup 2018, hạng nhì King's Cup 2019.
Phong cách chơi bóng:
•	Tốc độ nhanh, thể lực tốt, tranh chấp mạnh mẽ.
•	Khả năng lên công về thủ toàn diện, tạt bóng chuẩn xác.
•	Có tinh thần thi đấu quyết liệt, luôn nỗ lực hết mình.
Thành tích:
•	1 AFF Cup
•	1 HCV SEA Games
•	1 Cầu thủ trẻ xuất sắc nhất Đông Nam Á 2018
•	2 lần Quả bóng vàng U19 Việt Nam (2016, 2017)
Đoàn Văn Hậu là một trong những cầu thủ trẻ triển vọng nhất Việt Nam và được kỳ vọng sẽ trở thành trụ cột của ĐTQG trong tương lai.
"""
  print("""
Đoàn Văn Hậu (sinh ngày 19 tháng 4 năm 1999) là cầu thủ bóng đá chuyên nghiệp người Việt Nam chơi ở vị trí hậu vệ trái cho câu lạc bộ Hà Nội và Đội tuyển quốc gia Việt Nam.
Sự nghiệp:
•	Khởi nghiệp tại lò đào tạo trẻ Hà Nội T&T.
•	Nổi tiếng với biệt danh "Hậu tủ" và là một trong những hậu vệ trái xuất sắc nhất Việt Nam.
•	Thi đấu ấn tượng tại V.League 1, AFC Champions League và các giải đấu quốc tế cùng ĐTQG Việt Nam.
•	Chuyển sang SC Heerenveen (Hà Lan) vào năm 2019 nhưng gặp chấn thương và trở về Việt Nam thi đấu cho Hà Nội FC.
•	Góp công vào thành công của ĐTQG Việt Nam: vô địch AFF Cup 2018, hạng nhì King's Cup 2019.
Phong cách chơi bóng:
•	Tốc độ nhanh, thể lực tốt, tranh chấp mạnh mẽ.
•	Khả năng lên công về thủ toàn diện, tạt bóng chuẩn xác.
•	Có tinh thần thi đấu quyết liệt, luôn nỗ lực hết mình.
Thành tích:
•	1 AFF Cup
•	1 HCV SEA Games
•	1 Cầu thủ trẻ xuất sắc nhất Đông Nam Á 2018
•	2 lần Quả bóng vàng U19 Việt Nam (2016, 2017)
Đoàn Văn Hậu là một trong những cầu thủ trẻ triển vọng nhất Việt Nam và được kỳ vọng sẽ trở thành trụ cột của ĐTQG trong tương lai.
""")
if A == 'Erling Haaland':
  B = """
Erling Haaland (sinh ngày 21 tháng 7 năm 2000) là một cầu thủ bóng đá chuyên nghiệp người Na Uy đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Manchester City tại Premier League và Đội tuyển quốc gia Na Uy.
Sự nghiệp:
•	Khởi nghiệp tại Bryne FK (Na Uy) và Molde FK (Na Uy).
•	Nổi tiếng với biệt danh "Người ngoài hành tinh" và là một trong những tiền đạo trẻ xuất sắc nhất thế giới hiện nay.
•	Chuyển sang Red Bull Salzburg (Áo) vào năm 2018 và ghi được nhiều bàn thắng ấn tượng.
•	Gia nhập Borussia Dortmund (Đức) vào năm 2020 và tiếp tục tỏa sáng với khả năng ghi bàn bùng nổ.
•	Chuyển đến Manchester City vào năm 2023.
•	Thành viên ĐTQG Na Uy từ năm 2019.
Phong cách chơi bóng:
•	Tốc độ nhanh, thể lực sung mãn, sức mạnh dồi dào.
•	Khả năng dứt điểm đa dạng, thường xuyên ghi bàn đẹp mắt.
•	Chơi đầu tốt, di chuyển thông minh và có tầm nhìn chiến thuật.
Thành tích:
•	2 Cúp Quốc gia Áo
•	1 Cúp Đức
•	1 UEFA Champions League (Á quân)
•	1 Giải thưởng Cầu thủ trẻ xuất sắc nhất châu Âu 2020
Erling Haaland là một trong những cầu thủ trẻ xuất sắc nhất thế giới hiện nay và được kỳ vọng sẽ trở thành một trong những tiền đạo vĩ đại nhất trong tương lai.
"""
  print("""
Erling Haaland (sinh ngày 21 tháng 7 năm 2000) là một cầu thủ bóng đá chuyên nghiệp người Na Uy đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Manchester City tại Premier League và Đội tuyển quốc gia Na Uy.
Sự nghiệp:
•	Khởi nghiệp tại Bryne FK (Na Uy) và Molde FK (Na Uy).
•	Nổi tiếng với biệt danh "Người ngoài hành tinh" và là một trong những tiền đạo trẻ xuất sắc nhất thế giới hiện nay.
•	Chuyển sang Red Bull Salzburg (Áo) vào năm 2018 và ghi được nhiều bàn thắng ấn tượng.
•	Gia nhập Borussia Dortmund (Đức) vào năm 2020 và tiếp tục tỏa sáng với khả năng ghi bàn bùng nổ.
•	Chuyển đến Manchester City vào năm 2023.
•	Thành viên ĐTQG Na Uy từ năm 2019.
Phong cách chơi bóng:
•	Tốc độ nhanh, thể lực sung mãn, sức mạnh dồi dào.
•	Khả năng dứt điểm đa dạng, thường xuyên ghi bàn đẹp mắt.
•	Chơi đầu tốt, di chuyển thông minh và có tầm nhìn chiến thuật.
Thành tích:
•	2 Cúp Quốc gia Áo
•	1 Cúp Đức
•	1 UEFA Champions League (Á quân)
•	1 Giải thưởng Cầu thủ trẻ xuất sắc nhất châu Âu 2020
Erling Haaland là một trong những cầu thủ trẻ xuất sắc nhất thế giới hiện nay và được kỳ vọng sẽ trở thành một trong những tiền đạo vĩ đại nhất trong tương lai.
""")
if A == 'Gareth Bale':
  B = """
Gareth Bale (sinh ngày 16 tháng 10 năm 1989) là cầu thủ bóng đá chuyên nghiệp người Wales đang thi đấu ở vị trí tiền vệ cánh hoặc tiền đạo cho câu lạc bộ Los Angeles FC tại Major League Soccer và Đội tuyển quốc gia Wales.
Sự nghiệp:
•	Khởi nghiệp tại Southampton và sau đó chuyển sang Tottenham Hotspur.
•	Nổi tiếng với biệt danh "Cầu thủ chạy nhanh nhất thế giới" và là một trong những cầu thủ tấn công xuất sắc nhất thế hệ của anh ấy.
•	Chuyển đến Real Madrid vào năm 2013 với mức phí chuyển nhượng kỷ lục thế giới tại thời điểm đó.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 4 chức vô địch UEFA Champions League, 2 La Liga và 1 Copa del Rey.
•	Quay trở lại Tottenham Hotspur theo dạng cho mượn vào năm 2020 trước khi chuyển đến Los Angeles FC vào năm 2022.
•	Thành viên ĐTQG Wales từ năm 2006.
Phong cách chơi bóng:
•	Tốc độ nhanh, kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo.
•	Khả năng dứt điểm chính xác bằng cả hai chân, đặc biệt là những cú sút xa uy lực.
•	Chơi bóng thông minh, có khả năng kiến tạo và ghi bàn tốt.
Thành tích:
•	4 UEFA Champions League
•	2 La Liga
•	1 Copa del Rey
•	1 Cúp FA
•	1 League Cup
•	1 UEFA European Championship (Á quân)
Gareth Bale là một trong những cầu thủ xuất sắc nhất thế giới trong thập kỷ qua và là niềm tự hào của bóng đá Wales.
"""
  print("""
Gareth Bale (sinh ngày 16 tháng 10 năm 1989) là cầu thủ bóng đá chuyên nghiệp người Wales đang thi đấu ở vị trí tiền vệ cánh hoặc tiền đạo cho câu lạc bộ Los Angeles FC tại Major League Soccer và Đội tuyển quốc gia Wales.
Sự nghiệp:
•	Khởi nghiệp tại Southampton và sau đó chuyển sang Tottenham Hotspur.
•	Nổi tiếng với biệt danh "Cầu thủ chạy nhanh nhất thế giới" và là một trong những cầu thủ tấn công xuất sắc nhất thế hệ của anh ấy.
•	Chuyển đến Real Madrid vào năm 2013 với mức phí chuyển nhượng kỷ lục thế giới tại thời điểm đó.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 4 chức vô địch UEFA Champions League, 2 La Liga và 1 Copa del Rey.
•	Quay trở lại Tottenham Hotspur theo dạng cho mượn vào năm 2020 trước khi chuyển đến Los Angeles FC vào năm 2022.
•	Thành viên ĐTQG Wales từ năm 2006.
Phong cách chơi bóng:
•	Tốc độ nhanh, kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo.
•	Khả năng dứt điểm chính xác bằng cả hai chân, đặc biệt là những cú sút xa uy lực.
•	Chơi bóng thông minh, có khả năng kiến tạo và ghi bàn tốt.
Thành tích:
•	4 UEFA Champions League
•	2 La Liga
•	1 Copa del Rey
•	1 Cúp FA
•	1 League Cup
•	1 UEFA European Championship (Á quân)
Gareth Bale là một trong những cầu thủ xuất sắc nhất thế giới trong thập kỷ qua và là niềm tự hào của bóng đá Wales.
""")
if A == 'Jude Bellingham':
  B ="""
Jude Bellingham (sinh ngày 29 tháng 6 năm 2002) là cầu thủ bóng đá chuyên nghiệp người Anh đang thi đấu ở vị trí tiền vệ trung tâm cho câu lạc bộ Borussia Dortmund tại Bundesliga và Đội tuyển quốc gia Anh.
Sự nghiệp:
•	Khởi nghiệp tại Birmingham City và sau đó chuyển sang Borussia Dortmund vào năm 2020.
•	Nổi tiếng với biệt danh "Bellingham" và được đánh giá là một trong những tiền vệ trẻ triển vọng nhất thế giới hiện nay.
•	Chơi ấn tượng tại Bundesliga và Champions League, thu hút sự chú ý của nhiều câu lạc bộ lớn.
•	Thành viên ĐTQG Anh từ năm 2020 và tham dự EURO 2020.
Phong cách chơi bóng:
•	Khả năng chuyền bóng tốt, tầm nhìn chiến thuật rộng, nhãn quan chiến thuật sắc bén.
•	Chuyền bóng ngắn và dài đều tốt, có khả năng kiến tạo và ghi bàn.
•	Khả năng tranh chấp tay đôi tốt, di chuyển thông minh và có tinh thần thi đấu nhiệt huyết.
Thành tích:
•	1 DFB-Pokal
•	1 UEFA Europa League (Á quân)
Jude Bellingham là một trong những tiền vệ trẻ tài năng nhất thế giới hiện nay và được kỳ vọng sẽ trở thành một trong những ngôi sao hàng đầu trong tương lai.
"""
  print("""
Jude Bellingham (sinh ngày 29 tháng 6 năm 2002) là cầu thủ bóng đá chuyên nghiệp người Anh đang thi đấu ở vị trí tiền vệ trung tâm cho câu lạc bộ Borussia Dortmund tại Bundesliga và Đội tuyển quốc gia Anh.
Sự nghiệp:
•	Khởi nghiệp tại Birmingham City và sau đó chuyển sang Borussia Dortmund vào năm 2020.
•	Nổi tiếng với biệt danh "Bellingham" và được đánh giá là một trong những tiền vệ trẻ triển vọng nhất thế giới hiện nay.
•	Chơi ấn tượng tại Bundesliga và Champions League, thu hút sự chú ý của nhiều câu lạc bộ lớn.
•	Thành viên ĐTQG Anh từ năm 2020 và tham dự EURO 2020.
Phong cách chơi bóng:
•	Khả năng chuyền bóng tốt, tầm nhìn chiến thuật rộng, nhãn quan chiến thuật sắc bén.
•	Chuyền bóng ngắn và dài đều tốt, có khả năng kiến tạo và ghi bàn.
•	Khả năng tranh chấp tay đôi tốt, di chuyển thông minh và có tinh thần thi đấu nhiệt huyết.
Thành tích:
•	1 DFB-Pokal
•	1 UEFA Europa League (Á quân)
Jude Bellingham là một trong những tiền vệ trẻ tài năng nhất thế giới hiện nay và được kỳ vọng sẽ trở thành một trong những ngôi sao hàng đầu trong tương lai.
""")
if A == 'Karim Benzema':
  B ="""
Karim Benzema (sinh ngày 19 tháng 12 năm 1987) là cầu thủ bóng đá chuyên nghiệp người Pháp gốc Algérie đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Real Madrid tại La Liga và Đội tuyển quốc gia Pháp.
Sự nghiệp:
•	Khởi nghiệp tại Olympique Lyonnais và sau đó chuyển sang Real Madrid vào năm 2009.
•	Nổi tiếng với biệt danh "Benz" và là một trong những tiền đạo xuất sắc nhất thế giới hiện nay.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 5 UEFA Champions League, 4 La Liga và 2 Copa del Rey.
•	Quay trở lại ĐTQG Pháp sau 6 năm vắng bóng vào năm 2021 và góp công giúp Pháp vô địch UEFA Nations League 2021.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo, dứt điểm đa dạng.
•	Chơi bóng thông minh, có khả năng kiến tạo và ghi bàn tốt.
•	Di chuyển thông minh, phối hợp nhịp nhàng với đồng đội.
Thành tích:
•	5 UEFA Champions League
•	4 La Liga
•	2 Copa del Rey
•	1 UEFA European Championship
•	1 UEFA Nations League
Karim Benzema là một trong những cầu thủ xuất sắc nhất thế giới hiện nay và là huyền thoại của Real Madrid.
"""
  print("""
Karim Benzema (sinh ngày 19 tháng 12 năm 1987) là cầu thủ bóng đá chuyên nghiệp người Pháp gốc Algérie đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Real Madrid tại La Liga và Đội tuyển quốc gia Pháp.
Sự nghiệp:
•	Khởi nghiệp tại Olympique Lyonnais và sau đó chuyển sang Real Madrid vào năm 2009.
•	Nổi tiếng với biệt danh "Benz" và là một trong những tiền đạo xuất sắc nhất thế giới hiện nay.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 5 UEFA Champions League, 4 La Liga và 2 Copa del Rey.
•	Quay trở lại ĐTQG Pháp sau 6 năm vắng bóng vào năm 2021 và góp công giúp Pháp vô địch UEFA Nations League 2021.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo, dứt điểm đa dạng.
•	Chơi bóng thông minh, có khả năng kiến tạo và ghi bàn tốt.
•	Di chuyển thông minh, phối hợp nhịp nhàng với đồng đội.
Thành tích:
•	5 UEFA Champions League
•	4 La Liga
•	2 Copa del Rey
•	1 UEFA European Championship
•	1 UEFA Nations League
Karim Benzema là một trong những cầu thủ xuất sắc nhất thế giới hiện nay và là huyền thoại của Real Madrid.
""")
if A == 'Kevin De Bruyne':
  B = """
Kevin De Bruyne (sinh ngày 28 tháng 6 năm 1991) là cầu thủ bóng đá chuyên nghiệp người Bỉ đang thi đấu ở vị trí tiền vệ trung tâm cho câu lạc bộ Manchester City tại Premier League và đội tuyển quốc gia Bỉ. Được coi là một trong những tiền vệ xuất sắc nhất thế giới hiện nay, De Bruyne được biết đến với khả năng chuyền bóng, kiến tạo, tầm nhìn chiến thuật và kỹ thuật cá nhân điêu luyện.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại K.A.A. Gent và sau đó gia nhập KRC Genk vào năm 2005.
•	Chuyển đến Chelsea vào năm 2012 nhưng không thành công và được cho mượn đến Werder Bremen và VfL Wolfsburg.
•	Nổi tiếng tại Wolfsburg và được Manchester City mua vào năm 2015.
•	Gặt hái nhiều thành công cùng Man City, bao gồm 4 Premier League, 1 FA Cup và 8 League Cup.
•	Thành viên ĐTQG Bỉ từ năm 2010 và là đội trưởng hiện tại.
•	Tham dự World Cup 2014, Euro 2016, World Cup 2018 và Euro 2020, cùng đội tuyển Bỉ giành hạng ba World Cup 2018.
Phong cách chơi bóng:
•	Khả năng chuyền bóng tầm xa chính xác, tạo ra cơ hội ghi bàn cho đồng đội.
•	Chuyền bóng ngắn và dài đều tốt, có khả năng kiến tạo và ghi bàn.
•	Kỹ thuật cá nhân điêu luyện, rê bóng tốt, di chuyển thông minh.
Thành tích:
•	4 Premier League
•	1 FA Cup
•	8 League Cup
•	1 Cúp Bỉ
•	1 Hạng ba World Cup
Kevin De Bruyne là một trong những tiền vệ xuất sắc nhất thế giới hiện nay và là trụ cột của Manchester City và ĐTQG Bỉ.
"""
  print("""
Kevin De Bruyne (sinh ngày 28 tháng 6 năm 1991) là cầu thủ bóng đá chuyên nghiệp người Bỉ đang thi đấu ở vị trí tiền vệ trung tâm cho câu lạc bộ Manchester City tại Premier League và đội tuyển quốc gia Bỉ. Được coi là một trong những tiền vệ xuất sắc nhất thế giới hiện nay, De Bruyne được biết đến với khả năng chuyền bóng, kiến tạo, tầm nhìn chiến thuật và kỹ thuật cá nhân điêu luyện.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại K.A.A. Gent và sau đó gia nhập KRC Genk vào năm 2005.
•	Chuyển đến Chelsea vào năm 2012 nhưng không thành công và được cho mượn đến Werder Bremen và VfL Wolfsburg.
•	Nổi tiếng tại Wolfsburg và được Manchester City mua vào năm 2015.
•	Gặt hái nhiều thành công cùng Man City, bao gồm 4 Premier League, 1 FA Cup và 8 League Cup.
•	Thành viên ĐTQG Bỉ từ năm 2010 và là đội trưởng hiện tại.
•	Tham dự World Cup 2014, Euro 2016, World Cup 2018 và Euro 2020, cùng đội tuyển Bỉ giành hạng ba World Cup 2018.
Phong cách chơi bóng:
•	Khả năng chuyền bóng tầm xa chính xác, tạo ra cơ hội ghi bàn cho đồng đội.
•	Chuyền bóng ngắn và dài đều tốt, có khả năng kiến tạo và ghi bàn.
•	Kỹ thuật cá nhân điêu luyện, rê bóng tốt, di chuyển thông minh.
Thành tích:
•	4 Premier League
•	1 FA Cup
•	8 League Cup
•	1 Cúp Bỉ
•	1 Hạng ba World Cup
Kevin De Bruyne là một trong những tiền vệ xuất sắc nhất thế giới hiện nay và là trụ cột của Manchester City và ĐTQG Bỉ.
""")
if A == 'Kylian Mbappé':
  B = """
Kylian Mbappé Lottin (sinh ngày 20 tháng 12 năm 1998) là cầu thủ bóng đá chuyên nghiệp người Pháp đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Paris Saint-Germain tại Ligue 1 và đội tuyển quốc gia Pháp. Được mệnh danh là "Niềm hy vọng mới của bóng đá Pháp", Mbappé nổi tiếng với tốc độ bứt phá, kỹ thuật cá nhân điêu luyện và khả năng dứt điểm đa dạng.
Sự nghiệp:
•	Khởi nghiệp tại AS Bondy và sau đó gia nhập AS Monaco vào năm 2013.
•	Nổi tiếng tại Monaco và được Paris Saint-Germain mua vào năm 2018 với mức phí chuyển nhượng kỷ lục thế giới tại thời điểm đó dành cho cầu thủ trẻ.
•	Gặt hái nhiều thành công cùng PSG, bao gồm 4 Ligue 1, 4 Coupe de France và 2 Coupe de la Ligue.
•	Thành viên ĐTQG Pháp từ năm 2017 và là nhà vô địch World Cup 2018.
•	Tham dự World Cup 2018, Euro 2020 và UEFA Nations League 2021.
Phong cách chơi bóng:
•	Tốc độ bứt phá nhanh như chớp, khả năng rê bóng lắt léo, di chuyển thông minh.
•	Kỹ thuật cá nhân điêu luyện, dứt điểm đa dạng bằng cả hai chân.
•	Khả năng kiến tạo tốt, tạo ra cơ hội ghi bàn cho đồng đội.
Thành tích:
•	1 World Cup
•	1 UEFA Nations League
•	4 Ligue 1
•	4 Coupe de France
•	2 Coupe de la Ligue
Kylian Mbappé là một trong những cầu thủ trẻ xuất sắc nhất thế giới hiện nay và được kỳ vọng sẽ trở thành một trong những huyền thoại bóng đá Pháp trong tương lai.
"""
  print("""
Kylian Mbappé Lottin (sinh ngày 20 tháng 12 năm 1998) là cầu thủ bóng đá chuyên nghiệp người Pháp đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Paris Saint-Germain tại Ligue 1 và đội tuyển quốc gia Pháp. Được mệnh danh là "Niềm hy vọng mới của bóng đá Pháp", Mbappé nổi tiếng với tốc độ bứt phá, kỹ thuật cá nhân điêu luyện và khả năng dứt điểm đa dạng.
Sự nghiệp:
•	Khởi nghiệp tại AS Bondy và sau đó gia nhập AS Monaco vào năm 2013.
•	Nổi tiếng tại Monaco và được Paris Saint-Germain mua vào năm 2018 với mức phí chuyển nhượng kỷ lục thế giới tại thời điểm đó dành cho cầu thủ trẻ.
•	Gặt hái nhiều thành công cùng PSG, bao gồm 4 Ligue 1, 4 Coupe de France và 2 Coupe de la Ligue.
•	Thành viên ĐTQG Pháp từ năm 2017 và là nhà vô địch World Cup 2018.
•	Tham dự World Cup 2018, Euro 2020 và UEFA Nations League 2021.
Phong cách chơi bóng:
•	Tốc độ bứt phá nhanh như chớp, khả năng rê bóng lắt léo, di chuyển thông minh.
•	Kỹ thuật cá nhân điêu luyện, dứt điểm đa dạng bằng cả hai chân.
•	Khả năng kiến tạo tốt, tạo ra cơ hội ghi bàn cho đồng đội.
Thành tích:
•	1 World Cup
•	1 UEFA Nations League
•	4 Ligue 1
•	4 Coupe de France
•	2 Coupe de la Ligue
Kylian Mbappé là một trong những cầu thủ trẻ xuất sắc nhất thế giới hiện nay và được kỳ vọng sẽ trở thành một trong những huyền thoại bóng đá Pháp trong tương lai.
""")
if A == 'Luka Modrić':
  B = """
Luka Modrić (sinh ngày 9 tháng 9 năm 1985) là cầu thủ bóng đá chuyên nghiệp người Croatia đang thi đấu ở vị trí tiền vệ trung tâm cho câu lạc bộ Real Madrid tại La Liga và đội tuyển quốc gia Croatia. Được xem là một trong những tiền vệ xuất sắc nhất thế giới hiện nay, Modrić được biết đến với khả năng chuyền bóng, rê bóng, tầm nhìn chiến thuật và khả năng ghi bàn.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Dinamo Zagreb và sau đó gia nhập Tottenham Hotspur vào năm 2008.
•	Chuyển đến Real Madrid vào năm 2012 và gặt hái nhiều thành công vang dội, bao gồm 5 UEFA Champions League, 3 La Liga và 1 Copa del Rey.
•	Thành viên ĐTQG Croatia từ năm 2006 và là đội trưởng hiện tại.
•	Dẫn dắt Croatia giành hạng nhì World Cup 2018 và UEFA Nations League 2018.
•	Nhận giải Quả bóng Vàng 2018, Cầu thủ xuất sắc nhất châu Âu 2018 và The Best FIFA Men's Player 2018.
Phong cách chơi bóng:
•	Khả năng chuyền bóng tầm xa chính xác, tạo ra cơ hội ghi bàn cho đồng đội.
•	Chuyền bóng ngắn và dài đều tốt, có khả năng kiến tạo và ghi bàn.
•	Kỹ thuật cá nhân điêu luyện, rê bóng tốt, di chuyển thông minh.
Thành tích:
•	5 UEFA Champions League
•	3 La Liga
•	1 Copa del Rey
•	1 Á quân World Cup
•	1 UEFA Nations League
•	1 Quả bóng Vàng
•	1 Cầu thủ xuất sắc nhất châu Âu
•	1 The Best FIFA Men's Player
Luka Modrić là một trong những tiền vệ xuất sắc nhất thế giới hiện nay và là huyền thoại của Real Madrid và ĐTQG Croatia.
"""
  print("""
Luka Modrić (sinh ngày 9 tháng 9 năm 1985) là cầu thủ bóng đá chuyên nghiệp người Croatia đang thi đấu ở vị trí tiền vệ trung tâm cho câu lạc bộ Real Madrid tại La Liga và đội tuyển quốc gia Croatia. Được xem là một trong những tiền vệ xuất sắc nhất thế giới hiện nay, Modrić được biết đến với khả năng chuyền bóng, rê bóng, tầm nhìn chiến thuật và khả năng ghi bàn.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Dinamo Zagreb và sau đó gia nhập Tottenham Hotspur vào năm 2008.
•	Chuyển đến Real Madrid vào năm 2012 và gặt hái nhiều thành công vang dội, bao gồm 5 UEFA Champions League, 3 La Liga và 1 Copa del Rey.
•	Thành viên ĐTQG Croatia từ năm 2006 và là đội trưởng hiện tại.
•	Dẫn dắt Croatia giành hạng nhì World Cup 2018 và UEFA Nations League 2018.
•	Nhận giải Quả bóng Vàng 2018, Cầu thủ xuất sắc nhất châu Âu 2018 và The Best FIFA Men's Player 2018.
Phong cách chơi bóng:
•	Khả năng chuyền bóng tầm xa chính xác, tạo ra cơ hội ghi bàn cho đồng đội.
•	Chuyền bóng ngắn và dài đều tốt, có khả năng kiến tạo và ghi bàn.
•	Kỹ thuật cá nhân điêu luyện, rê bóng tốt, di chuyển thông minh.
Thành tích:
•	5 UEFA Champions League
•	3 La Liga
•	1 Copa del Rey
•	1 Á quân World Cup
•	1 UEFA Nations League
•	1 Quả bóng Vàng
•	1 Cầu thủ xuất sắc nhất châu Âu
•	1 The Best FIFA Men's Player
Luka Modrić là một trong những tiền vệ xuất sắc nhất thế giới hiện nay và là huyền thoại của Real Madrid và ĐTQG Croatia.
""")
if A == 'Messi':
  B = """
Lionel Andrés Messi (sinh ngày 24 tháng 6 năm 1987) là cầu thủ bóng đá chuyên nghiệp người Argentina đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Paris Saint-Germain tại Ligue 1 và đội tuyển quốc gia Argentina. Được mệnh danh là "El Pulga" (Chú Bọ), "La Pulga Atómica" (Chú Bọ Nguyên tử) và "El Mesías" (Đấng Cứu Thế), Messi được nhiều người đánh giá là cầu thủ xuất sắc nhất mọi thời đại.
Sự nghiệp:
•	Khởi nghiệp tại Newell's Old Boys và sau đó gia nhập FC Barcelona vào năm 2000.
•	Gặt hái nhiều thành công vang dội cùng Barcelona, bao gồm 10 La Liga, 4 UEFA Champions League và 7 Copa del Rey.
•	Chuyển đến Paris Saint-Germain vào năm 2021.
•	Thành viên ĐTQG Argentina từ năm 2005 và là đội trưởng hiện tại.
•	Dẫn dắt Argentina giành chức vô địch Copa América 2021 và Finalissima 2022.
•	Nhận giải Quả bóng Vàng kỷ lục 7 lần, Cầu thủ xuất sắc nhất châu Âu 4 lần và The Best FIFA Men's Player 1 lần.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, dứt điểm đa dạng.
•	Khả năng kiến tạo tuyệt vời, tạo ra cơ hội ghi bàn cho đồng đội.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, chơi bóng sáng tạo.
Thành tích:
•	10 La Liga
•	4 UEFA Champions League
•	7 Copa del Rey
•	1 Copa América
•	1 Finalissima
•	7 Quả bóng Vàng
•	4 Cầu thủ xuất sắc nhất châu Âu
•	1 The Best FIFA Men's Player
Lionel Messi là cầu thủ xuất sắc nhất mọi thời đại và là huyền thoại của FC Barcelona và ĐTQG Argentina.
"""
  print("""
Lionel Andrés Messi (sinh ngày 24 tháng 6 năm 1987) là cầu thủ bóng đá chuyên nghiệp người Argentina đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Paris Saint-Germain tại Ligue 1 và đội tuyển quốc gia Argentina. Được mệnh danh là "El Pulga" (Chú Bọ), "La Pulga Atómica" (Chú Bọ Nguyên tử) và "El Mesías" (Đấng Cứu Thế), Messi được nhiều người đánh giá là cầu thủ xuất sắc nhất mọi thời đại.
Sự nghiệp:
•	Khởi nghiệp tại Newell's Old Boys và sau đó gia nhập FC Barcelona vào năm 2000.
•	Gặt hái nhiều thành công vang dội cùng Barcelona, bao gồm 10 La Liga, 4 UEFA Champions League và 7 Copa del Rey.
•	Chuyển đến Paris Saint-Germain vào năm 2021.
•	Thành viên ĐTQG Argentina từ năm 2005 và là đội trưởng hiện tại.
•	Dẫn dắt Argentina giành chức vô địch Copa América 2021 và Finalissima 2022.
•	Nhận giải Quả bóng Vàng kỷ lục 7 lần, Cầu thủ xuất sắc nhất châu Âu 4 lần và The Best FIFA Men's Player 1 lần.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, dứt điểm đa dạng.
•	Khả năng kiến tạo tuyệt vời, tạo ra cơ hội ghi bàn cho đồng đội.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, chơi bóng sáng tạo.
Thành tích:
•	10 La Liga
•	4 UEFA Champions League
•	7 Copa del Rey
•	1 Copa América
•	1 Finalissima
•	7 Quả bóng Vàng
•	4 Cầu thủ xuất sắc nhất châu Âu
•	1 The Best FIFA Men's Player
Lionel Messi là cầu thủ xuất sắc nhất mọi thời đại và là huyền thoại của FC Barcelona và ĐTQG Argentina.
""")
if A == 'Mohamed Salah':
  B = """
Mohamed Salah Hamed Mahrous Ghaly (sinh ngày 15 tháng 6 năm 1992), hay còn được biết đến với biệt danh "Mo Salah", là cầu thủ bóng đá chuyên nghiệp người Ai Cập hiện đang thi đấu ở vị trí tiền đạo cánh phải cho câu lạc bộ Liverpool tại Premier League và đội trưởng của đội tuyển quốc gia Ai Cập.
Sự nghiệp:
•	Khởi nghiệp tại El Mokawloon và sau đó chuyển đến Basel vào năm 2013.
•	Nổi tiếng tại Basel và được Chelsea mua vào năm 2014 nhưng không thành công.
•	Được cho mượn đến Fiorentina và Roma, sau đó Roma mua đứt vào năm 2016.
•	Chuyển đến Liverpool vào năm 2017 và gặt hái nhiều thành công, bao gồm 1 Premier League, 1 UEFA Champions League và 1 FA Cup.
•	Thành viên ĐTQG Ai Cập từ năm 2011 và là đội trưởng hiện tại.
•	Dẫn dắt Ai Cập tham dự 2 World Cup (2018 và 2022) và 2 AFCON (2017 và 2021).
Phong cách chơi bóng:
•	Tốc độ bứt phá nhanh, kỹ thuật cá nhân điêu luyện, rê bóng lắt léo.
•	Khả năng dứt điểm đa dạng, ghi bàn bằng cả hai chân.
•	Chuyền bóng tốt, tạo ra cơ hội cho đồng đội.
•	Di chuyển thông minh, thi đấu nhiệt huyết.
Thành tích:
•	1 Premier League
•	1 UEFA Champions League
•	1 FA Cup
•	2 Giải thưởng Cầu thủ xuất sắc nhất châu Phi
•	1 Cầu thủ ghi bàn xuất sắc nhất Premier League
Mohamed Salah là một trong những cầu thủ cánh xuất sắc nhất thế giới hiện nay và là trụ cột của Liverpool và ĐTQG Ai Cập.
"""
  print("""
Mohamed Salah Hamed Mahrous Ghaly (sinh ngày 15 tháng 6 năm 1992), hay còn được biết đến với biệt danh "Mo Salah", là cầu thủ bóng đá chuyên nghiệp người Ai Cập hiện đang thi đấu ở vị trí tiền đạo cánh phải cho câu lạc bộ Liverpool tại Premier League và đội trưởng của đội tuyển quốc gia Ai Cập.
Sự nghiệp:
•	Khởi nghiệp tại El Mokawloon và sau đó chuyển đến Basel vào năm 2013.
•	Nổi tiếng tại Basel và được Chelsea mua vào năm 2014 nhưng không thành công.
•	Được cho mượn đến Fiorentina và Roma, sau đó Roma mua đứt vào năm 2016.
•	Chuyển đến Liverpool vào năm 2017 và gặt hái nhiều thành công, bao gồm 1 Premier League, 1 UEFA Champions League và 1 FA Cup.
•	Thành viên ĐTQG Ai Cập từ năm 2011 và là đội trưởng hiện tại.
•	Dẫn dắt Ai Cập tham dự 2 World Cup (2018 và 2022) và 2 AFCON (2017 và 2021).
Phong cách chơi bóng:
•	Tốc độ bứt phá nhanh, kỹ thuật cá nhân điêu luyện, rê bóng lắt léo.
•	Khả năng dứt điểm đa dạng, ghi bàn bằng cả hai chân.
•	Chuyền bóng tốt, tạo ra cơ hội cho đồng đội.
•	Di chuyển thông minh, thi đấu nhiệt huyết.
Thành tích:
•	1 Premier League
•	1 UEFA Champions League
•	1 FA Cup
•	2 Giải thưởng Cầu thủ xuất sắc nhất châu Phi
•	1 Cầu thủ ghi bàn xuất sắc nhất Premier League
Mohamed Salah là một trong những cầu thủ cánh xuất sắc nhất thế giới hiện nay và là trụ cột của Liverpool và ĐTQG Ai Cập.
""")
if A == 'Neymar':
  B = """
Neymar da Silva Santos Júnior (sinh ngày 5 tháng 2 năm 1992), thường được gọi là "Neymar" hoặc "Neymar Jr.", là cầu thủ bóng đá chuyên nghiệp người Brazil hiện đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Paris Saint-Germain tại Ligue 1 và đội tuyển quốc gia Brazil. Được mệnh danh là "Phù thủy bóng đá", Neymar nổi tiếng với kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo, tốc độ bứt phá nhanh và khả năng dứt điểm đa dạng.
Sự nghiệp:
•	Khởi nghiệp tại Santos FC và sau đó gia nhập FC Barcelona vào năm 2013.
•	Gặt hái nhiều thành công vang dội cùng Barcelona, bao gồm 2 La Liga, 1 UEFA Champions League và 3 Copa del Rey.
•	Chuyển đến Paris Saint-Germain vào năm 2017 với mức phí chuyển nhượng kỷ lục thế giới tại thời điểm đó.
•	Thành viên ĐTQG Brazil từ năm 2010 và là đội trưởng hiện tại.
•	Dẫn dắt Brazil giành huy chương vàng Olympic 2016 và Copa América 2019.
•	Nhận giải Quả bóng Bạc FIFA World Cup 2014.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, tốc độ bứt phá nhanh.
•	Khả năng dứt điểm đa dạng, ghi bàn bằng cả hai chân.
•	Chuyền bóng tốt, tạo ra cơ hội cho đồng đội.
•	Di chuyển thông minh, thi đấu sáng tạo.
Thành tích:
•	2 La Liga
•	1 UEFA Champions League
•	3 Copa del Rey
•	1 Copa América
•	1 Huy chương vàng Olympic
•	1 Quả bóng Bạc FIFA World Cup
Neymar Jr. là một trong những cầu thủ xuất sắc nhất thế giới hiện nay và là huyền thoại của FC Barcelona và ĐTQG Brazil.
"""
  print("""
Neymar da Silva Santos Júnior (sinh ngày 5 tháng 2 năm 1992), thường được gọi là "Neymar" hoặc "Neymar Jr.", là cầu thủ bóng đá chuyên nghiệp người Brazil hiện đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Paris Saint-Germain tại Ligue 1 và đội tuyển quốc gia Brazil. Được mệnh danh là "Phù thủy bóng đá", Neymar nổi tiếng với kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo, tốc độ bứt phá nhanh và khả năng dứt điểm đa dạng.
Sự nghiệp:
•	Khởi nghiệp tại Santos FC và sau đó gia nhập FC Barcelona vào năm 2013.
•	Gặt hái nhiều thành công vang dội cùng Barcelona, bao gồm 2 La Liga, 1 UEFA Champions League và 3 Copa del Rey.
•	Chuyển đến Paris Saint-Germain vào năm 2017 với mức phí chuyển nhượng kỷ lục thế giới tại thời điểm đó.
•	Thành viên ĐTQG Brazil từ năm 2010 và là đội trưởng hiện tại.
•	Dẫn dắt Brazil giành huy chương vàng Olympic 2016 và Copa América 2019.
•	Nhận giải Quả bóng Bạc FIFA World Cup 2014.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, tốc độ bứt phá nhanh.
•	Khả năng dứt điểm đa dạng, ghi bàn bằng cả hai chân.
•	Chuyền bóng tốt, tạo ra cơ hội cho đồng đội.
•	Di chuyển thông minh, thi đấu sáng tạo.
Thành tích:
•	2 La Liga
•	1 UEFA Champions League
•	3 Copa del Rey
•	1 Copa América
•	1 Huy chương vàng Olympic
•	1 Quả bóng Bạc FIFA World Cup
Neymar Jr. là một trong những cầu thủ xuất sắc nhất thế giới hiện nay và là huyền thoại của FC Barcelona và ĐTQG Brazil.
""")
if A == 'Nguyễn Hoàn Đức':
  B = """
Nguyễn Hoàng Đức (sinh ngày 14 tháng 1 năm 1998) là cầu thủ bóng đá chuyên nghiệp người Việt Nam đang thi đấu ở vị trí tiền vệ trung tâm cho câu lạc bộ Sông Lam Nghệ An tại V.League 1 và đội tuyển quốc gia Việt Nam. Được mệnh danh là "Hoàng tử Vàng của bóng đá Việt Nam", Hoàng Đức được biết đến với khả năng chuyền bóng, tắc bóng, tầm nhìn chiến thuật và tinh thần thi đấu quả cảm.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Viettel FC và sau đó gia nhập Sông Lam Nghệ An vào năm 2017.
•	Gặt hái nhiều thành công cùng Sông Lam Nghệ An, bao gồm 1 chức vô địch V.League 1 và 2 Cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2019 và là trụ cột không thể thay thế ở khu vực trung tuyến.
•	Dẫn dắt Việt Nam tham dự AFF Cup 2020 (á quân) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ trẻ xuất sắc nhất Việt Nam 2019 và 2020.
Phong cách chơi bóng:
•	Khả năng chuyền bóng chính xác, tạo ra cơ hội ghi bàn cho đồng đội.
•	Tắc bóng quyết đoán, tranh chấp mạnh mẽ, thu hồi bóng tốt.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, thi đấu hiệu quả.
•	Tinh thần thi đấu quả cảm, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	1 V.League 1
•	2 Cúp Quốc gia
•	1 Huy chương bạc AFF Cup
•	2 Giải Cầu thủ trẻ xuất sắc nhất Việt Nam
Nguyễn Hoàng Đức là một trong những tiền vệ trung tâm xuất sắc nhất Việt Nam hiện nay và là niềm hy vọng của bóng đá Việt Nam trong tương lai.
"""
  print("""
Nguyễn Hoàng Đức (sinh ngày 14 tháng 1 năm 1998) là cầu thủ bóng đá chuyên nghiệp người Việt Nam đang thi đấu ở vị trí tiền vệ trung tâm cho câu lạc bộ Sông Lam Nghệ An tại V.League 1 và đội tuyển quốc gia Việt Nam. Được mệnh danh là "Hoàng tử Vàng của bóng đá Việt Nam", Hoàng Đức được biết đến với khả năng chuyền bóng, tắc bóng, tầm nhìn chiến thuật và tinh thần thi đấu quả cảm.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Viettel FC và sau đó gia nhập Sông Lam Nghệ An vào năm 2017.
•	Gặt hái nhiều thành công cùng Sông Lam Nghệ An, bao gồm 1 chức vô địch V.League 1 và 2 Cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2019 và là trụ cột không thể thay thế ở khu vực trung tuyến.
•	Dẫn dắt Việt Nam tham dự AFF Cup 2020 (á quân) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ trẻ xuất sắc nhất Việt Nam 2019 và 2020.
Phong cách chơi bóng:
•	Khả năng chuyền bóng chính xác, tạo ra cơ hội ghi bàn cho đồng đội.
•	Tắc bóng quyết đoán, tranh chấp mạnh mẽ, thu hồi bóng tốt.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, thi đấu hiệu quả.
•	Tinh thần thi đấu quả cảm, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	1 V.League 1
•	2 Cúp Quốc gia
•	1 Huy chương bạc AFF Cup
•	2 Giải Cầu thủ trẻ xuất sắc nhất Việt Nam
Nguyễn Hoàng Đức là một trong những tiền vệ trung tâm xuất sắc nhất Việt Nam hiện nay và là niềm hy vọng của bóng đá Việt Nam trong tương lai.
""")
if A == 'Nguyễn Văn Quyết':
  B = """
Nguyễn Văn Quyết (sinh ngày 1 tháng 7 năm 1991) là cầu thủ bóng đá chuyên nghiệp người Việt Nam đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Hà Nội và đội tuyển quốc gia Việt Nam. Là đội trưởng của cả hai đội, Quyết được xem là một trong những cầu thủ xuất sắc nhất Việt Nam hiện nay và là biểu tượng của bóng đá Hà Nội và ĐTQG Việt Nam.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Hà Nội FC năm 2007 và gắn bó với đội bóng cho đến nay.
•	Gặt hái nhiều thành công cùng Hà Nội FC, bao gồm 6 chức vô địch V.League 1, 4 Cúp Quốc gia và 1 Siêu cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2012 và là đội trưởng hiện tại.
•	Dẫn dắt Việt Nam tham dự nhiều giải đấu quốc tế, bao gồm AFF Cup (vô địch 2018), Asian Cup (á quân 2019) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ xuất sắc nhất V.League 1 năm 2014 và Quả bóng Vàng Việt Nam 2 lần (2016 & 2017).
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, dứt điểm đa dạng.
•	Khả năng kiến tạo tốt, tạo ra cơ hội ghi bàn cho đồng đội.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, thi đấu sáng tạo.
•	Tinh thần thi đấu quả cảm, là đội trưởng mẫu mực, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	6 V.League 1
•	4 Cúp Quốc gia
•	1 Siêu cúp Quốc gia
•	1 AFF Cup
•	1 Quả bóng Vàng Việt Nam
•	1 Cầu thủ xuất sắc nhất V.League 1
Nguyễn Văn Quyết là một trong những cầu thủ xuất sắc nhất Việt Nam hiện nay và là huyền thoại của Hà Nội FC và ĐTQG Việt Nam.
"""
  print("""
Nguyễn Văn Quyết (sinh ngày 1 tháng 7 năm 1991) là cầu thủ bóng đá chuyên nghiệp người Việt Nam đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Hà Nội và đội tuyển quốc gia Việt Nam. Là đội trưởng của cả hai đội, Quyết được xem là một trong những cầu thủ xuất sắc nhất Việt Nam hiện nay và là biểu tượng của bóng đá Hà Nội và ĐTQG Việt Nam.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Hà Nội FC năm 2007 và gắn bó với đội bóng cho đến nay.
•	Gặt hái nhiều thành công cùng Hà Nội FC, bao gồm 6 chức vô địch V.League 1, 4 Cúp Quốc gia và 1 Siêu cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2012 và là đội trưởng hiện tại.
•	Dẫn dắt Việt Nam tham dự nhiều giải đấu quốc tế, bao gồm AFF Cup (vô địch 2018), Asian Cup (á quân 2019) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ xuất sắc nhất V.League 1 năm 2014 và Quả bóng Vàng Việt Nam 2 lần (2016 & 2017).
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, dứt điểm đa dạng.
•	Khả năng kiến tạo tốt, tạo ra cơ hội ghi bàn cho đồng đội.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, thi đấu sáng tạo.
•	Tinh thần thi đấu quả cảm, là đội trưởng mẫu mực, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	6 V.League 1
•	4 Cúp Quốc gia
•	1 Siêu cúp Quốc gia
•	1 AFF Cup
•	1 Quả bóng Vàng Việt Nam
•	1 Cầu thủ xuất sắc nhất V.League 1
Nguyễn Văn Quyết là một trong những cầu thủ xuất sắc nhất Việt Nam hiện nay và là huyền thoại của Hà Nội FC và ĐTQG Việt Nam.
""")
if A == 'Quang Hải':
  B = """
Nguyễn Quang Hải (sinh ngày 12 tháng 4 năm 1997) là cầu thủ bóng đá chuyên nghiệp người Việt Nam đang thi đấu ở vị trí tiền vệ tấn công cho câu lạc bộ Pau FC tại Ligue 2 và đội tuyển quốc gia Việt Nam. Được mệnh danh là "Hậu vệ trái xuất sắc nhất châu Á", Quang Hải được biết đến với khả năng rê bóng, chuyền bóng, kiến tạo và ghi bàn.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Hà Nội FC năm 2014 và thi đấu cho đội bóng đến năm 2022.
•	Gặt hái nhiều thành công cùng Hà Nội FC, bao gồm 4 chức vô địch V.League 1, 3 Cúp Quốc gia và 1 Siêu cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2016 và là trụ cột không thể thay thế ở khu vực hành lang cánh trái.
•	Dẫn dắt Việt Nam tham dự nhiều giải đấu quốc tế, bao gồm AFF Cup (vô địch 2018), Asian Cup (á quân 2019) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ trẻ xuất sắc nhất V.League 1 năm 2016, Quả bóng Vàng Việt Nam năm 2018 và Giải thưởng Cầu thủ xuất sắc nhất Đông Nam Á năm 2019.
Phong cách chơi bóng:
•	Khả năng rê bóng lắt léo, điêu luyện, thường xuyên tạo ra những pha đột phá nguy hiểm.
•	Chuyền bóng chính xác, tầm nhìn chiến thuật rộng, kiến tạo nhiều cơ hội cho đồng đội.
•	Khả năng dứt điểm tốt, ghi bàn bằng cả hai chân.
•	Di chuyển thông minh, thi đấu sáng tạo, có tinh thần đồng đội cao.
Thành tích:
•	4 V.League 1
•	3 Cúp Quốc gia
•	1 Siêu cúp Quốc gia
•	1 AFF Cup
•	1 Quả bóng Vàng Việt Nam
•	1 Giải thưởng Cầu thủ xuất sắc nhất Đông Nam Á
Nguyễn Quang Hải là một trong những cầu thủ xuất sắc nhất Việt Nam hiện nay và là niềm tự hào của bóng đá Việt Nam.
"""
  print("""
Nguyễn Quang Hải (sinh ngày 12 tháng 4 năm 1997) là cầu thủ bóng đá chuyên nghiệp người Việt Nam đang thi đấu ở vị trí tiền vệ tấn công cho câu lạc bộ Pau FC tại Ligue 2 và đội tuyển quốc gia Việt Nam. Được mệnh danh là "Hậu vệ trái xuất sắc nhất châu Á", Quang Hải được biết đến với khả năng rê bóng, chuyền bóng, kiến tạo và ghi bàn.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Hà Nội FC năm 2014 và thi đấu cho đội bóng đến năm 2022.
•	Gặt hái nhiều thành công cùng Hà Nội FC, bao gồm 4 chức vô địch V.League 1, 3 Cúp Quốc gia và 1 Siêu cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2016 và là trụ cột không thể thay thế ở khu vực hành lang cánh trái.
•	Dẫn dắt Việt Nam tham dự nhiều giải đấu quốc tế, bao gồm AFF Cup (vô địch 2018), Asian Cup (á quân 2019) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ trẻ xuất sắc nhất V.League 1 năm 2016, Quả bóng Vàng Việt Nam năm 2018 và Giải thưởng Cầu thủ xuất sắc nhất Đông Nam Á năm 2019.
Phong cách chơi bóng:
•	Khả năng rê bóng lắt léo, điêu luyện, thường xuyên tạo ra những pha đột phá nguy hiểm.
•	Chuyền bóng chính xác, tầm nhìn chiến thuật rộng, kiến tạo nhiều cơ hội cho đồng đội.
•	Khả năng dứt điểm tốt, ghi bàn bằng cả hai chân.
•	Di chuyển thông minh, thi đấu sáng tạo, có tinh thần đồng đội cao.
Thành tích:
•	4 V.League 1
•	3 Cúp Quốc gia
•	1 Siêu cúp Quốc gia
•	1 AFF Cup
•	1 Quả bóng Vàng Việt Nam
•	1 Giải thưởng Cầu thủ xuất sắc nhất Đông Nam Á
Nguyễn Quang Hải là một trong những cầu thủ xuất sắc nhất Việt Nam hiện nay và là niềm tự hào của bóng đá Việt Nam.
""")
if A == 'Roberto Carlos':
  B ="""
Roberto Carlos da Silva Rocha (sinh ngày 19 tháng 4 năm 1973), thường được gọi là "Roberto Carlos" hoặc đơn giản là "Carlos", là cựu cầu thủ bóng đá chuyên nghiệp người Brazil thi đấu ở vị trí hậu vệ trái. Được coi là một trong những hậu vệ trái xuất sắc nhất mọi thời đại, Carlos nổi tiếng với cú sút phạt mạnh mẽ với tốc độ và độ xoáy cao, khả năng rê bóng và kiến tạo ấn tượng.
Sự nghiệp:
•	Khởi nghiệp tại União Suzano và sau đó gia nhập Inter Milan vào năm 1991.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 4 La Liga, 3 Champions League và 2 Intercontinental Cup.
•	Thành viên ĐTQG Brazil từ năm 1995 đến năm 2006 và là trụ cột không thể thay thế ở vị trí hậu vệ trái.
•	Dẫn dắt Brazil vô địch World Cup 2002 và Copa América 1999 & 2002.
•	Nhận giải Quả bóng Vàng FIFA năm 2002 và được FIFA vinh danh là một trong 100 cầu thủ vĩ đại nhất mọi thời đại.
Phong cách chơi bóng:
•	Cú sút phạt mạnh mẽ với tốc độ và độ xoáy cao, ghi bàn nhiều từ những tình huống cố định.
•	Khả năng rê bóng lắt léo, tốc độ bứt phá nhanh, thường xuyên tham gia tấn công.
•	Khả năng tạt bóng chính xác, tạo ra cơ hội ghi bàn cho đồng đội.
•	Kỹ thuật cá nhân điêu luyện, thi đấu thông minh, có tinh thần đồng đội cao.
Thành tích:
•	4 La Liga
•	3 Champions League
•	2 Intercontinental Cup
•	1 World Cup
•	2 Copa América
•	1 Quả bóng Vàng FIFA
Roberto Carlos là một trong những hậu vệ trái xuất sắc nhất mọi thời đại và là huyền thoại của Real Madrid và ĐTQG Brazil.
"""
  print("""
Roberto Carlos da Silva Rocha (sinh ngày 19 tháng 4 năm 1973), thường được gọi là "Roberto Carlos" hoặc đơn giản là "Carlos", là cựu cầu thủ bóng đá chuyên nghiệp người Brazil thi đấu ở vị trí hậu vệ trái. Được coi là một trong những hậu vệ trái xuất sắc nhất mọi thời đại, Carlos nổi tiếng với cú sút phạt mạnh mẽ với tốc độ và độ xoáy cao, khả năng rê bóng và kiến tạo ấn tượng.
Sự nghiệp:
•	Khởi nghiệp tại União Suzano và sau đó gia nhập Inter Milan vào năm 1991.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 4 La Liga, 3 Champions League và 2 Intercontinental Cup.
•	Thành viên ĐTQG Brazil từ năm 1995 đến năm 2006 và là trụ cột không thể thay thế ở vị trí hậu vệ trái.
•	Dẫn dắt Brazil vô địch World Cup 2002 và Copa América 1999 & 2002.
•	Nhận giải Quả bóng Vàng FIFA năm 2002 và được FIFA vinh danh là một trong 100 cầu thủ vĩ đại nhất mọi thời đại.
Phong cách chơi bóng:
•	Cú sút phạt mạnh mẽ với tốc độ và độ xoáy cao, ghi bàn nhiều từ những tình huống cố định.
•	Khả năng rê bóng lắt léo, tốc độ bứt phá nhanh, thường xuyên tham gia tấn công.
•	Khả năng tạt bóng chính xác, tạo ra cơ hội ghi bàn cho đồng đội.
•	Kỹ thuật cá nhân điêu luyện, thi đấu thông minh, có tinh thần đồng đội cao.
Thành tích:
•	4 La Liga
•	3 Champions League
•	2 Intercontinental Cup
•	1 World Cup
•	2 Copa América
•	1 Quả bóng Vàng FIFA
Roberto Carlos là một trong những hậu vệ trái xuất sắc nhất mọi thời đại và là huyền thoại của Real Madrid và ĐTQG Brazil.
""")
if A == 'Rodrygo Goes':
  B = """
Rodrygo Silva de Goes, thường được gọi là "Rodrygo" hoặc "Rodrygo Goes", sinh ngày 19 tháng 1 năm 2001, là cầu thủ bóng đá chuyên nghiệp người Brazil hiện đang thi đấu ở vị trí tiền đạo cánh phải và tiền đạo trung tâm cho câu lạc bộ Real Madrid tại La Liga và đội tuyển quốc gia Brazil. Được mệnh danh là "Vinicius 2.0", Rodrygo nổi tiếng với kỹ thuật cá nhân điêu luyện, tốc độ bứt phá nhanh, khả năng dứt điểm đa dạng và tinh thần thi đấu nhiệt huyết.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Santos FC và gia nhập Real Madrid vào năm 2019.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 2 La Liga, 1 Champions League và 1 UEFA Super Cup.
•	Thành viên ĐTQG Brazil từ năm 2019 và là một trong những cầu thủ trẻ triển vọng nhất của bóng đá Brazil.
•	Tham dự Olympic Mùa hè 2020 cùng đội tuyển Olympic Brazil và giành huy chương vàng.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, tốc độ bứt phá nhanh.
•	Khả năng dứt điểm đa dạng, ghi bàn bằng cả hai chân.
•	Di chuyển thông minh, thi đấu sáng tạo, thường xuyên tạo ra những pha đột phá nguy hiểm.
•	Tinh thần thi đấu nhiệt huyết, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	2 La Liga
•	1 Champions League
•	1 UEFA Super Cup
•	1 Huy chương vàng Olympic
•	1 Giải thưởng Cầu thủ trẻ xuất sắc nhất Nam Mỹ 2019
Rodrygo Goes là một trong những cầu thủ trẻ triển vọng nhất thế giới hiện nay và là niềm hy vọng của bóng đá Brazil trong tương lai.
"""
  print("""
Rodrygo Silva de Goes, thường được gọi là "Rodrygo" hoặc "Rodrygo Goes", sinh ngày 19 tháng 1 năm 2001, là cầu thủ bóng đá chuyên nghiệp người Brazil hiện đang thi đấu ở vị trí tiền đạo cánh phải và tiền đạo trung tâm cho câu lạc bộ Real Madrid tại La Liga và đội tuyển quốc gia Brazil. Được mệnh danh là "Vinicius 2.0", Rodrygo nổi tiếng với kỹ thuật cá nhân điêu luyện, tốc độ bứt phá nhanh, khả năng dứt điểm đa dạng và tinh thần thi đấu nhiệt huyết.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Santos FC và gia nhập Real Madrid vào năm 2019.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 2 La Liga, 1 Champions League và 1 UEFA Super Cup.
•	Thành viên ĐTQG Brazil từ năm 2019 và là một trong những cầu thủ trẻ triển vọng nhất của bóng đá Brazil.
•	Tham dự Olympic Mùa hè 2020 cùng đội tuyển Olympic Brazil và giành huy chương vàng.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, tốc độ bứt phá nhanh.
•	Khả năng dứt điểm đa dạng, ghi bàn bằng cả hai chân.
•	Di chuyển thông minh, thi đấu sáng tạo, thường xuyên tạo ra những pha đột phá nguy hiểm.
•	Tinh thần thi đấu nhiệt huyết, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	2 La Liga
•	1 Champions League
•	1 UEFA Super Cup
•	1 Huy chương vàng Olympic
•	1 Giải thưởng Cầu thủ trẻ xuất sắc nhất Nam Mỹ 2019
Rodrygo Goes là một trong những cầu thủ trẻ triển vọng nhất thế giới hiện nay và là niềm hy vọng của bóng đá Brazil trong tương lai.
""")
if A == 'Ronaldinho':
  B = """
Ronaldo de Assis Moreira, thường được gọi là "Ronaldinho" hoặc "R10", sinh ngày 21 tháng 3 năm 1980, là cựu cầu thủ bóng đá chuyên nghiệp người Brazil thi đấu ở vị trí tiền vệ tấn công và tiền đạo. Được mệnh danh là "Phù thủy bóng đá", Ronaldinho nổi tiếng với kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo, những pha sút phạt uốn lượn và lối chơi đầy sáng tạo.
Sự nghiệp:
•	Khởi nghiệp tại Grêmio và sau đó gia nhập Paris Saint-Germain vào năm 2001.
•	Gặt hái nhiều thành công cùng Barcelona, bao gồm 2 La Liga, 1 Champions League và 2 Copa del Rey.
•	Thành viên ĐTQG Brazil từ năm 1999 đến năm 2013 và là biểu tượng của bóng đá Brazil trong thời kỳ hoàng kim.
•	Dẫn dắt Brazil vô địch World Cup 2002 và Copa América 2004 & 2007.
•	Nhận giải Quả bóng Vàng FIFA năm 2005 và được FIFA vinh danh là một trong 100 cầu thủ vĩ đại nhất mọi thời đại.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, thường xuyên tạo ra những pha đột phá đẹp mắt.
•	Khả năng sút phạt uốn lượn, ghi bàn từ những tình huống cố định.
•	Chuyền bóng chính xác, tầm nhìn chiến thuật rộng, kiến tạo nhiều cơ hội cho đồng đội.
•	Lối chơi đầy sáng tạo, mang đến niềm vui cho khán giả.
Thành tích:
•	2 La Liga
•	1 Champions League
•	2 Copa del Rey
•	1 World Cup
•	2 Copa América
•	1 Quả bóng Vàng FIFA
Ronaldinho là một trong những cầu thủ vĩ đại nhất mọi thời đại và là huyền thoại của Barcelona và ĐTQG Brazil.
"""
  print("""
Ronaldo de Assis Moreira, thường được gọi là "Ronaldinho" hoặc "R10", sinh ngày 21 tháng 3 năm 1980, là cựu cầu thủ bóng đá chuyên nghiệp người Brazil thi đấu ở vị trí tiền vệ tấn công và tiền đạo. Được mệnh danh là "Phù thủy bóng đá", Ronaldinho nổi tiếng với kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo, những pha sút phạt uốn lượn và lối chơi đầy sáng tạo.
Sự nghiệp:
•	Khởi nghiệp tại Grêmio và sau đó gia nhập Paris Saint-Germain vào năm 2001.
•	Gặt hái nhiều thành công cùng Barcelona, bao gồm 2 La Liga, 1 Champions League và 2 Copa del Rey.
•	Thành viên ĐTQG Brazil từ năm 1999 đến năm 2013 và là biểu tượng của bóng đá Brazil trong thời kỳ hoàng kim.
•	Dẫn dắt Brazil vô địch World Cup 2002 và Copa América 2004 & 2007.
•	Nhận giải Quả bóng Vàng FIFA năm 2005 và được FIFA vinh danh là một trong 100 cầu thủ vĩ đại nhất mọi thời đại.
Phong cách chơi bóng:
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, thường xuyên tạo ra những pha đột phá đẹp mắt.
•	Khả năng sút phạt uốn lượn, ghi bàn từ những tình huống cố định.
•	Chuyền bóng chính xác, tầm nhìn chiến thuật rộng, kiến tạo nhiều cơ hội cho đồng đội.
•	Lối chơi đầy sáng tạo, mang đến niềm vui cho khán giả.
Thành tích:
•	2 La Liga
•	1 Champions League
•	2 Copa del Rey
•	1 World Cup
•	2 Copa América
•	1 Quả bóng Vàng FIFA
Ronaldinho là một trong những cầu thủ vĩ đại nhất mọi thời đại và là huyền thoại của Barcelona và ĐTQG Brazil.
""")
if A == 'Ronaldo':
  B ="""
Cristiano Ronaldo dos Santos Aveiro, thường được gọi là "CR7", sinh ngày 5 tháng 2 năm 1985, là cầu thủ bóng đá chuyên nghiệp người Bồ Đào Nha đang thi đấu ở vị trí tiền đạo và đội trưởng của câu lạc bộ Al-Nassr tại Saudi Pro League và đội tuyển quốc gia Bồ Đào Nha. Được mệnh danh là "Cỗ máy ghi bàn", Ronaldo nổi tiếng với khả năng ghi bàn đa dạng, kỹ thuật cá nhân điêu luyện, ý chí quyết tâm cao và tinh thần thi đấu chuyên nghiệp.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Sporting CP và gia nhập Manchester United vào năm 2003.
•	Gặt hái nhiều thành công cùng Manchester United, bao gồm 3 Premier League, 1 Champions League và 1 FIFA Club World Cup.
•	Chuyển đến Real Madrid vào năm 2009 và trở thành cầu thủ ghi bàn nhiều nhất mọi thời đại của câu lạc bộ.
•	Cùng Real Madrid giành 4 Champions League, 2 La Liga và 2 Copa del Rey.
•	Thành viên ĐTQG Bồ Đào Nha từ năm 2003 và là cầu thủ khoác áo đội tuyển nhiều nhất.
•	Dẫn dắt Bồ Đào Nha vô địch Euro 2016 và UEFA Nations League 2018-19.
•	Nhận giải Quả bóng Vàng FIFA 5 lần và là cầu thủ ghi bàn nhiều nhất mọi thời đại của bóng đá quốc tế.
Phong cách chơi bóng:
•	Khả năng ghi bàn đa dạng, ghi bàn bằng cả hai chân và đầu.
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, tốc độ bứt phá nhanh.
•	Sức mạnh, khả năng tỳ đè tốt, tranh chấp bóng mạnh mẽ.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, thi đấu hiệu quả.
•	Ý chí quyết tâm cao, tinh thần thi đấu chuyên nghiệp, luôn đặt mục tiêu cao nhất.
Thành tích:
•	3 Premier League
•	4 Champions League
•	2 La Liga
•	2 Copa del Rey
•	1 Euro
•	1 UEFA Nations League
•	5 Quả bóng Vàng FIFA
Cristiano Ronaldo là một trong những cầu thủ vĩ đại nhất mọi thời đại và là huyền thoại của Manchester United, Real Madrid và ĐTQG Bồ Đào Nha.
"""
  print("""
Cristiano Ronaldo dos Santos Aveiro, thường được gọi là "CR7", sinh ngày 5 tháng 2 năm 1985, là cầu thủ bóng đá chuyên nghiệp người Bồ Đào Nha đang thi đấu ở vị trí tiền đạo và đội trưởng của câu lạc bộ Al-Nassr tại Saudi Pro League và đội tuyển quốc gia Bồ Đào Nha. Được mệnh danh là "Cỗ máy ghi bàn", Ronaldo nổi tiếng với khả năng ghi bàn đa dạng, kỹ thuật cá nhân điêu luyện, ý chí quyết tâm cao và tinh thần thi đấu chuyên nghiệp.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Sporting CP và gia nhập Manchester United vào năm 2003.
•	Gặt hái nhiều thành công cùng Manchester United, bao gồm 3 Premier League, 1 Champions League và 1 FIFA Club World Cup.
•	Chuyển đến Real Madrid vào năm 2009 và trở thành cầu thủ ghi bàn nhiều nhất mọi thời đại của câu lạc bộ.
•	Cùng Real Madrid giành 4 Champions League, 2 La Liga và 2 Copa del Rey.
•	Thành viên ĐTQG Bồ Đào Nha từ năm 2003 và là cầu thủ khoác áo đội tuyển nhiều nhất.
•	Dẫn dắt Bồ Đào Nha vô địch Euro 2016 và UEFA Nations League 2018-19.
•	Nhận giải Quả bóng Vàng FIFA 5 lần và là cầu thủ ghi bàn nhiều nhất mọi thời đại của bóng đá quốc tế.
Phong cách chơi bóng:
•	Khả năng ghi bàn đa dạng, ghi bàn bằng cả hai chân và đầu.
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, tốc độ bứt phá nhanh.
•	Sức mạnh, khả năng tỳ đè tốt, tranh chấp bóng mạnh mẽ.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, thi đấu hiệu quả.
•	Ý chí quyết tâm cao, tinh thần thi đấu chuyên nghiệp, luôn đặt mục tiêu cao nhất.
Thành tích:
•	3 Premier League
•	4 Champions League
•	2 La Liga
•	2 Copa del Rey
•	1 Euro
•	1 UEFA Nations League
•	5 Quả bóng Vàng FIFA
Cristiano Ronaldo là một trong những cầu thủ vĩ đại nhất mọi thời đại và là huyền thoại của Manchester United, Real Madrid và ĐTQG Bồ Đào Nha.
""")
if A == 'Tiến Dũng':
  B ="""
Bùi Tiến Dũng (sinh ngày 19 tháng 2 năm 1995) là cầu thủ bóng đá chuyên nghiệp người Việt Nam thi đấu ở vị trí trung vệ cho câu lạc bộ Viettel và đội tuyển quốc gia Việt Nam. Được mệnh danh là "dũng sĩ thép", Bùi Tiến Dũng được biết đến với khả năng tranh chấp mạnh mẽ, tinh thần thi đấu quả cảm và đóng góp quan trọng vào thành công của đội tuyển Việt Nam.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Viettel FC năm 2016 và gắn bó với đội bóng cho đến nay.
•	Gặt hái nhiều thành công cùng Viettel FC, bao gồm 2 chức vô địch V.League 1 và 1 Cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2018 và là trụ cột không thể thay thế ở khu vực trung tuyến.
•	Dẫn dắt Việt Nam tham dự nhiều giải đấu quốc tế, bao gồm AFF Cup (vô địch 2018), Asian Cup (á quân 2019) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ trẻ xuất sắc nhất Việt Nam năm 2018 và 2019.
Phong cách chơi bóng:
•	Khả năng tranh chấp mạnh mẽ, tắc bóng quyết đoán, thu hồi bóng tốt.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, thi đấu hiệu quả.
•	Tinh thần thi đấu quả cảm, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	2 V.League 1
•	1 Cúp Quốc gia
•	1 AFF Cup
•	2 Giải Cầu thủ trẻ xuất sắc nhất Việt Nam
"""
  print("""
Bùi Tiến Dũng (sinh ngày 19 tháng 2 năm 1995) là cầu thủ bóng đá chuyên nghiệp người Việt Nam thi đấu ở vị trí trung vệ cho câu lạc bộ Viettel và đội tuyển quốc gia Việt Nam. Được mệnh danh là "dũng sĩ thép", Bùi Tiến Dũng được biết đến với khả năng tranh chấp mạnh mẽ, tinh thần thi đấu quả cảm và đóng góp quan trọng vào thành công của đội tuyển Việt Nam.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Viettel FC năm 2016 và gắn bó với đội bóng cho đến nay.
•	Gặt hái nhiều thành công cùng Viettel FC, bao gồm 2 chức vô địch V.League 1 và 1 Cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2018 và là trụ cột không thể thay thế ở khu vực trung tuyến.
•	Dẫn dắt Việt Nam tham dự nhiều giải đấu quốc tế, bao gồm AFF Cup (vô địch 2018), Asian Cup (á quân 2019) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ trẻ xuất sắc nhất Việt Nam năm 2018 và 2019.
Phong cách chơi bóng:
•	Khả năng tranh chấp mạnh mẽ, tắc bóng quyết đoán, thu hồi bóng tốt.
•	Tầm nhìn chiến thuật rộng, di chuyển thông minh, thi đấu hiệu quả.
•	Tinh thần thi đấu quả cảm, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	2 V.League 1
•	1 Cúp Quốc gia
•	1 AFF Cup
•	2 Giải Cầu thủ trẻ xuất sắc nhất Việt Nam
""")
if A == 'Văn Toàn':
  B = """
Nguyễn Văn Toàn (sinh ngày 12 tháng 4 năm 1996) là cầu thủ bóng đá chuyên nghiệp người Việt Nam đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Hải Phòng và đội tuyển quốc gia Việt Nam. Được mệnh danh là "mũi tên vàng", Văn Toàn được biết đến với tốc độ bứt phá nhanh, khả năng rê bóng lắt léo và kỹ thuật dứt điểm sắc bén.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại HAGL năm 2016 và thi đấu cho đội bóng đến năm 2022.
•	Gặt hái nhiều thành công cùng HAGL, bao gồm 1 chức vô địch V.League 1 và 2 Cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2019 và là một trong những cầu thủ tấn công nguy hiểm nhất của đội tuyển.
•	Dẫn dắt Việt Nam tham dự nhiều giải đấu quốc tế, bao gồm AFF Cup (vô địch 2022) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ xuất sắc nhất V.League 1 năm 2020.
Phong cách chơi bóng:
•	Tốc độ bứt phá nhanh, di chuyển thông minh, thường xuyên tạo ra những pha đột phá nguy hiểm.
•	Khả năng rê bóng lắt léo, đi bóng qua người tốt.
•	Kỹ thuật dứt điểm sắc bén, ghi bàn bằng cả hai chân.
•	Tinh thần thi đấu nhiệt huyết, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	1 V.League 1
•	2 Cúp Quốc gia
•	1 AFF Cup
•	1 Giải Cầu thủ xuất sắc nhất V.League 1
Nguyễn Văn Toàn là một trong những cầu thủ tấn công xuất sắc nhất Việt Nam hiện nay và là niềm hy vọng của bóng đá Việt Nam trong tương lai.
"""
  print("""
Nguyễn Văn Toàn (sinh ngày 12 tháng 4 năm 1996) là cầu thủ bóng đá chuyên nghiệp người Việt Nam đang thi đấu ở vị trí tiền đạo cho câu lạc bộ Hải Phòng và đội tuyển quốc gia Việt Nam. Được mệnh danh là "mũi tên vàng", Văn Toàn được biết đến với tốc độ bứt phá nhanh, khả năng rê bóng lắt léo và kỹ thuật dứt điểm sắc bén.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại HAGL năm 2016 và thi đấu cho đội bóng đến năm 2022.
•	Gặt hái nhiều thành công cùng HAGL, bao gồm 1 chức vô địch V.League 1 và 2 Cúp Quốc gia.
•	Thành viên ĐTQG Việt Nam từ năm 2019 và là một trong những cầu thủ tấn công nguy hiểm nhất của đội tuyển.
•	Dẫn dắt Việt Nam tham dự nhiều giải đấu quốc tế, bao gồm AFF Cup (vô địch 2022) và Vòng loại thứ ba World Cup 2022.
•	Nhận giải Cầu thủ xuất sắc nhất V.League 1 năm 2020.
Phong cách chơi bóng:
•	Tốc độ bứt phá nhanh, di chuyển thông minh, thường xuyên tạo ra những pha đột phá nguy hiểm.
•	Khả năng rê bóng lắt léo, đi bóng qua người tốt.
•	Kỹ thuật dứt điểm sắc bén, ghi bàn bằng cả hai chân.
•	Tinh thần thi đấu nhiệt huyết, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	1 V.League 1
•	2 Cúp Quốc gia
•	1 AFF Cup
•	1 Giải Cầu thủ xuất sắc nhất V.League 1
Nguyễn Văn Toàn là một trong những cầu thủ tấn công xuất sắc nhất Việt Nam hiện nay và là niềm hy vọng của bóng đá Việt Nam trong tương lai.
""")
if A == 'Vinícius Júnior':
  B = """
Vinícius José Paixão de Oliveira Júnior, thường được gọi là "Vinícius Júnior" hoặc "Vini Jr", sinh ngày 12 tháng 10 năm 2000, là cầu thủ bóng đá chuyên nghiệp người Brazil thi đấu ở vị trí tiền đạo cánh trái cho câu lạc bộ Real Madrid tại La Liga và đội tuyển quốc gia Brazil. Được mệnh danh là "Ngôi sao mới của Real Madrid", Vinícius Júnior nổi tiếng với tốc độ bứt phá nhanh, kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo và những pha dứt điểm hiểm hóc.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Flamengo và gia nhập Real Madrid vào năm 2018.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 2 La Liga, 1 Champions League và 1 UEFA Super Cup.
•	Thành viên ĐTQG Brazil từ năm 2019 và là một trong những cầu thủ trẻ triển vọng nhất của bóng đá Brazil.
•	Tham dự Olympic Mùa hè 2020 cùng đội tuyển Olympic Brazil và giành huy chương vàng.
Phong cách chơi bóng:
•	Tốc độ bứt phá nhanh, thường xuyên tạo ra những pha đột phá nguy hiểm bên cánh trái.
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, qua người ấn tượng.
•	Khả năng dứt điểm hiểm hóc, ghi bàn bằng cả hai chân.
•	Di chuyển thông minh, thi đấu sáng tạo, thường xuyên tạo ra cơ hội ghi bàn cho đồng đội.
•	Tinh thần thi đấu nhiệt huyết, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	2 La Liga
•	1 Champions League
•	1 UEFA Super Cup
•	1 Huy chương vàng Olympic
•	1 Giải thưởng Cầu thủ trẻ xuất sắc nhất Nam Mỹ 2019
Vinícius Júnior là một trong những cầu thủ trẻ triển vọng nhất thế giới hiện nay và là niềm hy vọng của bóng đá Brazil trong tương lai.
"""
  print("""
Vinícius José Paixão de Oliveira Júnior, thường được gọi là "Vinícius Júnior" hoặc "Vini Jr", sinh ngày 12 tháng 10 năm 2000, là cầu thủ bóng đá chuyên nghiệp người Brazil thi đấu ở vị trí tiền đạo cánh trái cho câu lạc bộ Real Madrid tại La Liga và đội tuyển quốc gia Brazil. Được mệnh danh là "Ngôi sao mới của Real Madrid", Vinícius Júnior nổi tiếng với tốc độ bứt phá nhanh, kỹ thuật cá nhân điêu luyện, khả năng rê bóng lắt léo và những pha dứt điểm hiểm hóc.
Sự nghiệp:
•	Bắt đầu sự nghiệp tại Flamengo và gia nhập Real Madrid vào năm 2018.
•	Gặt hái nhiều thành công cùng Real Madrid, bao gồm 2 La Liga, 1 Champions League và 1 UEFA Super Cup.
•	Thành viên ĐTQG Brazil từ năm 2019 và là một trong những cầu thủ trẻ triển vọng nhất của bóng đá Brazil.
•	Tham dự Olympic Mùa hè 2020 cùng đội tuyển Olympic Brazil và giành huy chương vàng.
Phong cách chơi bóng:
•	Tốc độ bứt phá nhanh, thường xuyên tạo ra những pha đột phá nguy hiểm bên cánh trái.
•	Kỹ thuật cá nhân điêu luyện, rê bóng lắt léo, qua người ấn tượng.
•	Khả năng dứt điểm hiểm hóc, ghi bàn bằng cả hai chân.
•	Di chuyển thông minh, thi đấu sáng tạo, thường xuyên tạo ra cơ hội ghi bàn cho đồng đội.
•	Tinh thần thi đấu nhiệt huyết, không ngại va chạm, luôn đặt lợi ích tập thể lên hàng đầu.
Thành tích:
•	2 La Liga
•	1 Champions League
•	1 UEFA Super Cup
•	1 Huy chương vàng Olympic
•	1 Giải thưởng Cầu thủ trẻ xuất sắc nhất Nam Mỹ 2019
Vinícius Júnior là một trong những cầu thủ trẻ triển vọng nhất thế giới hiện nay và là niềm hy vọng của bóng đá Brazil trong tương lai.
""")
# Tiếng Anh
translator = Translator()
translation_en = translator.translate(text=B, dest='en')
print("English:", translation_en.text)
# Tiếng Hàn
translation_ko = translator.translate(text=B, dest='ko')
print("Korean:", translation_ko.text)
# Tiếng Nhật
translation_ja = translator.translate(text=B, dest='ja')
print("Japanese:", translation_ja.text)
# Tiếng Trung Phổn Thể
translation_zh_TW = translator.translate(text=B, dest='zh-tw')
print("Traditional Chinese:", translation_zh_TW.text)
# Tiếng Trung Giản Thể
translation_zh_CN = translator.translate(text=B, dest='zh-cn')
print("Simplified Chinese:", translation_zh_CN.text)
