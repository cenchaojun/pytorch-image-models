import pandas as pd


def deal():
    # 列表
    company_name_list = ['12312', '141', '515', '41']

    # list转dataframe
    df = pd.DataFrame(company_name_list)

    # 保存到本地excel
    df.to_csv("company_name_li.csv", index=False)


if __name__ == '__main__':
    deal()
