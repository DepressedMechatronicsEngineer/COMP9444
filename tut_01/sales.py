sales = "2543,4,4.34;5463,7,8.31;7765,10,1.23;9833,9,34.12;5056,4,5.67;7657,10,4.23;3343,7,2.98;3778,9,9.27;1118,5,8.23;3873,3,4.45;6588,2,5.67;5778,6,3.41;7765,11,2.23;9343,8,4.12;5057,5,4.67;7657,5,4.23;3356,7,4.98;3776,8,8.27;1228,5,7.23;3873,2,4.50"


total_sales_amount = 0.0
for sale in sales.split(";"):
    _, quantity, price = sale.split(",")
    total_sale = int(quantity) * float(price)

    total_sales_amount += total_sale

print(f"Total sales amount: ${total_sales_amount:.2f}")