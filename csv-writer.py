import csv

# header = ['car', 'color', 'volume', 'weight', 'co2']
# data = [
#     ['Honda Civic', 'red', 1600, 1252, 94],
#     ['Susuki Swift', 'white', 1300, '', 101],
#     ['Mazda 3', 'black', '', 1280, 104],
#     ['Benz CLA', 'black', '', 1465, 102],
#     ['Mini Cooper', 'red', 1500, '', 105],
# ]

# with open('test-missing-values.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)


# header = ['Weather', 'TimeOfWeek', 'TimeOfDay', 'TrafficJam']
# data = [
#         ['Clear', 'Workday', 'Morning', 'Yes'],
#         ['Clear', 'Workday', 'Evening', 'Yes'],
#         ['Clear', 'Weekend', 'Lunch', 'No'],
#         ['Rainy', 'Workday', 'Morning', 'Yes'],
#         ['Rainy', 'Workday', 'Lunch', 'Yes'],
#         ['Rainy', 'Workday', 'Evening', 'Yes'],
#         ['Rainy', 'Weekend', 'Lunch', 'No'],
#         ['Snowy', 'Workday', 'Morning', 'Yes'],
#         ['Snowy', 'Workday', 'Evening', 'Yes'],
#         ['Snowy', 'Weekend', 'Lunch', 'No'],
# ]

# with open('test-label-encoder.xlsx', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)


# header = ['car', 'color', 'volume', 'weight', 'co2']
# data = [
#     ['Honda Civic', 'red', 1.6, 1252, 94],
#     ['Susuki Swift', 'white', 1.3, 990, 101],
#     ['Mazda 3', 'black', 2.2, 1280, 104],
#     ['Benz CLA', 'black', 1.5, 1465, 102],
#     ['Mini Cooper', 'red', 1.5, 1140, 105],
#     ['Ford Focus', 'blue', 2.0, 1328, 105],
#     ['Benz C-Class', 'silver', 2.1, 1365, 99],
#     ['Benz E-Class', 'white', 2.1, 16.5, 115],
#     ['Ford Fiesta', 'red', 1.5, 1112, 98],
#     ['Volvo XC70', 'silver', 2.0, 1746, 117],
# ]

# with open('test-standard-scaler.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)



# header = ['month', 'rainfall_mm', 'umbrellas_sold']
# data = [
#     ['Jan', 82.0, 15],
#     ['Feb', 92.5, 25],
#     ['Mar', 83.2, 17],
#     ['Apr', 76.9, 14],
#     ['May', 84.0, 18],
#     ['Jun', 87.4, 19],
#     ['Jul', 90.1, 21],
#     ['Aug', 97.8, 24],
#     ['Sep', 103.2, 28],
#     ['Oct', 118.0, 36],
#     ['Nov', 91.2, 20],
#     ['Dec', 104.6, 32],
#     ['Jan', 84.2, 14],
#     ['Feb', 80.1, 11],
#     ['Mar', 75.0, 9],
#     ['Apr', 77.8, 12],
#     ['May', 82.0, 14],
#     ['Jun', 85.1, 18],
#     ['Jul', 86.5, 20],
#     ['Aug', 91.4, 23],
#     ['Sep', 96.7, 27],
#     ['Oct', 99.7, 29],
#     ['Nov', 105.4, 31],
#     ['Dec', 102.3, 31],
# ]

# with open('umbrellas_sold_1.xlsx', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)



# header = ['Gestational_Age_wks', 'Birth_Weight_gm']
# data = [
#     [34.6, 1895],
#     [36.0, 2030],
#     [29.3, 1440],
#     [36.5, 2340],
#     [40.2, 3120],
#     [38.0, 2680],
#     [37.5, 2500],
#     [36.0, 2120],
#     [39.2, 2560],
#     [40.0, 3010],
#     [37.6, 2750],
#     [35.0, 2250],
#     [37.4, 2475],
#     [41.1, 3260],
#     [38.0, 2680],
#     [38.7, 2005]
# ]

# with open('birth_weight.xlsx', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)

# header = ['interest_rate', 'unemployment_rate', 'stock_index_price']
# data = [
#     [2.75, 5.3, 1464],
#     [2.50, 5.3, 1394],
#     [2.50, 5.3, 1357],
#     [2.50, 5.4, 1302],
#     [2.50, 5.4, 1294],
#     [2.50, 5.4, 1270],
#     [2.25, 5.5, 1253],
#     [2.25, 5.5, 1201],
#     [2.25, 5.5, 1157],
#     [2.25, 5.6, 1134],
#     [2.25, 5.6, 1099],
#     [2.25, 5.7, 1025],
#     [2.25, 5.7, 989],
#     [2.00, 5.8, 979],
#     [2.00, 5.8, 952],
#     [2.00, 5.9, 921],
#     [2.00, 5.9, 898],
#     [1.75, 6.0, 853],
#     [1.75, 6.0, 814],
#     [1.75, 6.1, 787],
#     [1.75, 6.1, 747],
#     [1.75, 6.2, 704],
#     [1.75, 6.1, 719]
# ]

# with open('stock_index_price.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)


# header = ['car', 'model', 'volume', 'weight', 'co2']
# data = [
#     ['Toyota', 'Aygo', 1000, 790, 99],
#     ['Mitsubishi', 'Space Star', 1200, 1160, 95],
#     ['Skoda', 'Citigo', 1000, 929, 95],
#     ['Fiat', '500', 900, 865, 90],
#     ['Volvo', 'XC60', 2000, 3900, 170],
#     ['Buick', 'Enclave', 3600, 4500, 190],
#     ['Lexus', 'RX', 3500, 4300, 200],
#     ['Hyundai', 'Tucson', 2000, 3200, 130],
#     ['Kia', 'Optima', 2400, 3100, 140],
#     ['Jeep', 'Cherokee', 2400, 3700, 160],
#     ['Mazda', 'CX-5', 2500, 3200, 150],
#     ['Volkswagen', 'Passat', 2000, 3300, 145],
#     ['Subaru', 'Outback', 2500, 3500, 155],
#     ['Honda', 'Accord', 2000, 3200, 140],
#     ['Chevrolet', 'Equinox', 1500, 3300, 135],
#     ['Ford', 'Fusion', 2500, 3100, 145],
#     ['Audi', 'A4', 2000, 3200, 150],
#     ['Mercedes-Benz', 'E-Class', 3000, 3800, 170],
#     ['Hyundai', 'Sonata', 2400, 3000, 135],
#     ['Nissan', 'Altima', 2500, 3100, 140],
#     ['BMW', 'X5', 3000, 4000, 180],
#     ['Tesla', 'Model S', 0, 4500, 0],
#     ['Chevrolet', 'Malibu', 2000, 3300, 130],
#     ['Ford', 'Escape', 2000, 3500, 140],
#     ['Honda', 'Civic', 1800, 2800, 120],
#     ['Toyota', 'Camry', 2500, 3200, 150],
#     ['Ford', 'B-Max', 1600, 1235, 104],
#     ['BMW', '2', 1600, 1390, 108],
#     ['Opel', 'Zafira', 1600, 1405, 109],
#     ['Mercedes', 'SLK', 2500, 1395, 120],
#     ['Subaru', 'Forester', 2500, 3200, 145],
#     ['Chevrolet', 'Traverse', 3600, 4400, 190],
#     ['Ford', 'Explorer', 3000, 4200, 180],
#     ['Nissan', 'Rogue', 2500, 3500, 160],
#     ['Toyota', 'Rav4', 2500, 3400, 155],
#     ['Mercedes-Benz', 'GLC', 2.0, 3800, 170]

# ]

# with open('co2_emission.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

#     # write multiple rows
#     writer.writerows(data)


header = ['home_size', 'kilowatt_hours_per_month']
data = [[1290, 1182],
        [1350, 1172],
        [1470, 1264],
        [1650, 1470],
        [1850, 1550],
        [1970, 1610],
        [2100, 1800],
        [2230, 1840],
        [2400, 1956],
        [2930, 1954]
]

with open('electricity-consumption.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)