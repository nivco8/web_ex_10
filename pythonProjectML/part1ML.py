##imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Distribution for each variable
from pandas import DataFrame


def dist(feature, type, title, xlab, ylab, save_output_path):
    fig = plt.figure()
    if type == 'plot':
        plt.plot(feature, linewidth=3)
        plt.title(title, fontsize=20)
        plt.xlabel(xlab, fontsize=15)
        plt.ylabel(ylab, fontsize=15)
        fig.savefig(save_output_path)
    elif type == 'countPlot':
        sns.countplot(x=xlab, data=feature)
        sns.title(title, fontsize=20)
        fig.savefig(save_output_path)
    elif type == 'hist':
        plt.hist(feature, bins=30)
        plt.title(title, fontsize=20)
        plt.xlabel(xlab, fontsize=15)
        plt.ylabel(ylab, fontsize=15)
        fig.savefig(save_output_path)
    plt.show()


def corr_per_row(df: DataFrame):
    fig = plt.figure()
    corr_mat = np.zeros((len(df.columns), len(df.columns)))
    for col_i, col_1_name in enumerate(df.columns):
        for col_j, col_2_name in enumerate(df.columns):
            col1 = df[col_1_name]
            col2 = df[col_2_name]
            cols = pd.concat([col1, col2], axis=1)
            corr_mat[col_i, col_j] = cols.dropna().corr().iloc[0, 1]
    corr_df = pd.DataFrame(np.abs(corr_mat), index=df.columns, columns=df.columns) # .sort_values('target')
    sns.heatmap(corr_df, annot=True, cmap='coolwarm')
    plt.show()


def fisher_per_row(df: DataFrame):
    plt.figure()
    fisher_mat = np.zeros((len(df.columns) - 1))
    for col_i, col_1_name in enumerate(df.columns[:-1]):
        col_pos = df[col_1_name][df.target == 1]
        col_neg = df[col_1_name][df.target == 0]

        fisher = (np.abs(col_pos.dropna().mean() - col_neg.dropna().mean())) / (col_pos.dropna().std() + col_neg.dropna().std())
        fisher_mat[col_i] = fisher

    corr_df = pd.DataFrame(fisher_mat, index=df.columns[:-1], columns=["target"]).sort_values('target')
    sns.heatmap(corr_df, annot=True, cmap='coolwarm')
    plt.title("fisher")
    plt.show()


def analyze_categorical_feature(df, feature):
    plt.figure()
    if feature == 'city':
        df = df.sort_values("city_development_index")
    feature_col = df[feature].dropna()
    target_col = df.target[~df[feature].isna()]
    target_one_precentages = {}
    cadidates_per_cat = {}
    for cat in feature_col.unique():
        target_one_precentage = target_col[feature_col == cat].sum() / len(target_col[feature_col==cat])
        target_one_precentages[cat] = target_one_precentage
        cadidates_per_cat[cat] = (feature_col == cat).sum() / len(feature_col)
        cadidates_per_cat[cat] = (feature_col == cat).sum()


    plt.plot(list(target_one_precentages.keys()), list(target_one_precentages.values()), '-*')
    plt.bar(list(cadidates_per_cat.keys()), list(cadidates_per_cat.values()))

    plt.show()


def df_to_numeric(df: DataFrame) -> DataFrame:
    company_size_lut = {'1000-4999': 3000, '500-999': 750, '<10': 5, '50-99': 75, '10000+': 15000, '5000-9999': 7500, '100-500': 300, 'Oct-49': None}
    city_lut = {name: int(name.split('_')[-1]) for name in df.city.unique()}
    gender_lut = {"Male": 0, "Female": 1, "Other": 2}
    relevent_experience_lut = {'No relevent experience': 0, 'Has relevent experience': 1}
    enrolled_lut = {'no_enrollment': 0, 'Part time course': 1, 'Full time course':2}
    education_lut = {'High School': 1, 'Masters': 3, 'Graduate': 2, 'Phd': 4, 'Primary School': 0}
    major_lut = {'STEM': 4, 'Arts': 1, 'Other': 2, 'Humanities': 3, 'No Major': 0, 'Business Degree': 5}
    company_type_lut = {'Pvt Ltd': 0, 'Public Sector': 1, 'NGO': 2, 'Other': 3, 'Early Stage Startup': 4, 'Funded Startup': 5}
    last_new_job_lut = {'never': 0, '1': 1, '2': 2, '>4': 5, '3': 3, '4': 4}
    experience_lut = {name: int(name) for name in df.experience.unique() if not (pd.isna(name) or ("<" in name) or (">" in name))}
    experience_lut['>20'] = 21
    experience_lut['<1'] = 0

    df.company_size = df.company_size.replace(company_size_lut)
    df.city = df.city.replace(city_lut)
    df.gender = df.gender.replace(gender_lut)
    df.relevent_experience = df.relevent_experience.replace(relevent_experience_lut)
    df.enrolled_university = df.enrolled_university.replace(enrolled_lut)
    df.education_level = df.education_level.replace(education_lut)
    df.major_discipline = df.major_discipline.replace(major_lut)
    df.company_type = df.company_type.replace(company_type_lut)
    df.last_new_job = df.last_new_job.replace(last_new_job_lut)
    df.experience = df.experience.replace(experience_lut)

    return df


# def preprocess(df: DataFrame) -> DataFrame:


if _name_ == "_main_":
    XYtrain = pd.read_csv(
        r"C:\Users\shaul\Desktop\שאולי לימודים\2022\לימוד מכונה\פרוייקט משיין\DATA\XY_train.csv")  # Loading the Final Dataset From Part A:
    XYdescribe = XYtrain.describe()
    analyze_categorical_feature(XYtrain, "city")
    XYtrain_numeric = df_to_numeric(XYtrain)
    corr_per_row(XYtrain_numeric)
    fisher_per_row(XYtrain_numeric)
    dist(
        feature=sorted(XYtrain['city_development_index']),
        type='plot',
        title='city development',
        xlab='gender',
        ylab='count',
        save_output_path="a.png"
    )

    # f3 = dist(XYtrainNN['gender'], 'hist', 'Gender', 'gender', 'count')
    # f4 = dist(XYtrainNN['relevent_experience'], 'hist', 'Gender', 'gender', 'count')
    # f5 = dist(XYtrainNN['enrolled_university'], 'hist', 'Gender', 'gender', 'count')
    # f6 = dist(XYtrainNN['education_level'], 'hist', 'Gender', 'gender', 'count')
    # f7 = dist(XYtrainNN['major_discipline'], 'hist', 'Gender', 'gender', 'count')
    # f8 = dist(XYtrainNN['experience'], 'hist', 'Gender', 'gender', 'count')
    # f9 = dist(XYtrainNN['company_size'], 'hist', 'Gender', 'gender', 'count')
    # f10 = dist(XYtrainNN['company_typeb'], 'hist', 'Gender', 'gender', 'count')
    # f11 = dist(XYtrainNN['last_new_job'], 'hist', 'Gender', 'gender', 'count')
    # f12 = dist(XYtrainNN['training_hours'], 'hist', 'Gender', 'gender', 'count')
    # f13 = dist(XYtrainNN['target'], 'hist', 'Gender', 'gender', 'count')



#------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from django.utils.timezone import utc
import sqlite3 as sql1
import os as os
import datetime
from datetime import datetime
import sys
from sklearn.decomposition import PCA

# initializing


XYtrain = pd.read_csv('C:\ניב\לימודים\שנה ד\ML\פרויקט/XY_train.csv')
print(XYtrain.count)   #Count the number of samples in the Dataset



pd.set_option('display.float_format', lambda x: '%.2f' % x)
python_script_path = os.getcwd()
csv_path = os.path.join(python_script_path, r'XY_train.csv')
dataBase = pd.DataFrame(pd.read_csv(csv_path))
dataFixed = pd.DataFrame(index=range(0, 8999),
                         columns=['s_title', 's_publishedAt', 's_channelId', 's_channelTitle', 's_categoryId',
                                  's_trending_date', 's_tags', 's_view_count', 's_likes', 's_dislikes',
                                  's_comment_count', 's_description'])
dataErased = pd.DataFrame(index=range(0, 8999),
                          columns=['s_title', 's_publishedAt', 's_channelId', 's_channelTitle', 's_categoryId',
                                   's_trending_date', 's_tags', 's_view_count', 's_likes', 's_dislikes',
                                   's_comment_count', 's_description'])

DBExist = os.path.isfile('training.db')
if DBExist:
    os.remove('training.db')
_conn = sql1.connect('training.db')
_conn.text_factory = str
cur = _conn.cursor()


def main(arg):
    printDescribe(dataBase)
    printCategoryId_dist(dataBase)
    printView_count_dist(dataBase)
    printLikes_dist(dataBase)
    printDislikes_dist(dataBase)
    printComment_count_dist(dataBase)
    printCorr(dataBase)
    dislikes_VS_comment_count(dataBase)
    likes_VS_dislikes(dataBase)
    likes_VS_comment_count(dataBase)
    dislikes_VS_comment_count(dataBase)
    preProcessing(dataBase)
    ##REMOVING the rellevant values
    DB = dataBase
    X = DB.dropna(subset=['likes'])
    X.reset_index(drop=True, inplace=True)
    dataBase = X
    ####
    segmentation(dataBase)
    featureExtraction(dataBase)
    featureRepresentation(dataBase)
    featureSelection(dataBase)
    dimentionalityReduction(dataBase)


def dimentionalityReduction(db):
    data = dataBase.drop(
        ['title', 'publishedAt', 'channelId', 'channelTitle', 'trending_date', 'tags', 'description', 'tagsArray',
         'URLsArray', 'Trending', 'PublishDate', 'daysSinceLast', 'SerialNum', 'text'], 1)
    data = data.drop(
        ['view_count', 'onlyEnglish', 'categoryId', 'likesPace', 'dislikesPace', 'commentPace', 'daysSincePub',
         'isVEVO'], 1)
    pca = PCA(n_components=0.9)
    pca.fit(data)
    reduced = pca.transform(data)
    finalData = pd.DataFrame(reduced)
    finalData['view_count'] = dataBase['view_count']
    corrF = data.corr(method='pearson')
    print(finalData.describe())


def featureSelection(db):
    datatoSelect = dataBase.drop(
        ['title', 'publishedAt', 'channelId', 'channelTitle', 'trending_date', 'tags', 'description', 'tagsArray',
         'URLsArray', 'Trending', 'PublishDate', 'daysSinceLast', 'SerialNum', 'text'], 1)

    DBE = datatoSelect.drop(
        ['onlyEnglish', 'tagsAmount', 'URLAmount', 'likesPace', 'dislikesPace', 'commentPace', 'dislikesPerDay',
         'likesPerDay', 'commentCountPerDay', 'daysSincePub', 'dislikesSinceLast', 'likesSinceLast',
         'commentCountSinceLast', 'isVEVO'], 1)
    DBE1 = datatoSelect.drop(
        ['PublishHour', 'categoryId', 'likes', 'dislikes', 'comment_count', 'likesPace', 'dislikesPace', 'commentPace',
         'dislikesPerDay', 'likesPerDay', 'commentCountPerDay', 'daysSincePub', 'dislikesSinceLast', 'likesSinceLast',
         'commentCountSinceLast', 'isVEVO'], 1)
    DBE2 = datatoSelect.drop(
        ['PublishHour', 'categoryId', 'likes', 'dislikes', 'comment_count', 'dislikesPerDay', 'likesPerDay',
         'commentCountPerDay', 'commentCountPerDay', 'daysSincePub', 'dislikesSinceLast', 'likesSinceLast',
         'commentCountSinceLast', 'isVEVO'], 1)
    DBE3 = datatoSelect.drop(
        ['PublishHour', 'categoryId', 'likes', 'dislikes', 'comment_count', 'onlyEnglish', 'tagsAmount', 'URLAmount',
         'likesPace', 'dislikesPace', 'commentPace', 'daysSincePub', 'dislikesSinceLast', 'likesSinceLast',
         'commentCountSinceLast', 'isVEVO'], 1)
    DBE4 = datatoSelect.drop(
        ['PublishHour', 'categoryId', 'likes', 'dislikes', 'comment_count', 'onlyEnglish', 'tagsAmount', 'URLAmount',
         'likesPace', 'dislikesPace', 'commentPace', 'dislikesPerDay', 'likesPerDay', 'commentCountPerDay',
         'commentCountPerDay'], 1)

    # Pearson Correlation

    plt.figure(figsize=(12, 10))
    cor = DBE.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    # Correlation with view_count
    cor_target = abs(cor["view_count"])
    print(cor_target)
    # correlation of variables
    print(datatoSelect[["likes", "dislikes", 'comment_count', 'PublishHour']].corr())


def preProcessing(db):
    # dropTable()
    createTable()
    pasteData(db)
    dealMissingData(db)


def featureRepresentation(dataBase):
    URLAmount(dataBase)
    tagsAmount(dataBase)
    categorizeText(dataBase)
    DBres = dataBase.drop(
        ['view_count', 'title', 'publishedAt', 'channelId', 'channelTitle', 'trending_date', 'tags', 'description',
         'tagsArray', 'URLsArray', 'Trending', 'PublishDate', 'daysSinceLast', 'SerialNum', 'text', 'PublishHour',
         'categoryId', 'likes', 'dislikes', 'comment_count', 'likesPace', 'dislikesPace', 'commentPace',
         'dislikesPerDay', 'likesPerDay', 'commentCountPerDay', 'daysSincePub', 'dislikesSinceLast', 'likesSinceLast',
         'commentCountSinceLast', 'isVEVO'], 1)
    printDescribe(DBres)


def URLAmount(db):
    db.insert(loc=12, column='URLAmount', value='x')
    for i in range(0, len(db)):
        x = len(db['URLsArray'][i])
        db['URLAmount'].iat[i] = float(x)


def tagsAmount(db):
    db.insert(loc=12, column='tagsAmount', value='x')
    for i in range(0, len(db)):
        x = len(db['tagsArray'][i])
        db['tagsAmount'].iat[i] = float(x)


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def categorizeText(db):
    db.insert(loc=12, column='onlyEnglish', value='x')
    for i in range(0, len(db)):
        x = 1
        if not is_ascii(db['text'][i].strip()):
            x = 0
        db['onlyEnglish'].iat[i] = float(x)


def concatToString(db):
    db.insert(loc=12, column='text', value='x')
    for i in range(0, len(db)):
        x = db['description'][i] + db['channelTitle'][i] + db['title'][i] + db['tags'][i]
        db['text'].iat[i] = x


def isVEVO(db):
    db.insert(loc=12, column='isVEVO', value='x')
    for i in range(0, len(db)):
        if db['channelTitle'][i].find("VEVO") > 0:
            x = 1
        else:
            x = 0
        db['isVEVO'].iat[i] = int(x)


def fillSerial(Arr, stri, db):
    for i in range(0, len(db)):
        if stri == db['title'][i]:
            for j in range(0, len(Arr)):
                if Arr[j] == db['Trending'][i]:
                    db['SerialNum'].iat[i] = j + 1


def sampleSerialNum(db):
    db.insert(loc=12, column='SerialNum', value='x')
    Arr = []
    stri = db['title'][0]
    for i in range(0, len(db)):
        if stri == db['title'][i]:
            Arr.append(db['Trending'][i])
        else:
            Arr.sort()
            fillSerial(Arr, stri, db)
            Arr = []
            stri = db['title'][i]
            Arr.append(db['Trending'][i])
    Arr.sort()
    fillSerial(Arr, stri, db)


def reset():
    dataBase = pd.read_csv('C:/Users/amirt/Downloads/XY_train.csv')
    DB = dataBase
    X = DB.dropna(subset=['likes'])
    X.reset_index(drop=True, inplace=True)
    dataBase = X
    preProcessing(dataBase)
    segmentation(dataBase)
    sampleSerialNum(dataBase)


def calculateDays(x, y):
    date_format = '%Y-%m-%d'
    a = datetime.strptime(x, date_format)
    b = datetime.strptime(y, date_format)
    delta = a - b
    return delta.days


def daysBetweenSamples(db):
    dataBase.insert(loc=12, column='daysSinceLast', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['Trending'].iat[i]
                y = db['Trending'].iat[i + 1]
                db['daysSinceLast'].iat[i] = int(calculateDays(x, y))
            else:
                db['daysSinceLast'].iat[i] = int(0)
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['Trending'].iat[i]
                y = db['Trending'].iat[i + 1]
                db['daysSinceLast'].iat[i] = int(calculateDays(x, y))
            else:
                db['daysSinceLast'].iat[i] = int(0)


def commentCountBetweenSamples(db):
    dataBase.insert(loc=12, column='commentCountSinceLast', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['comment_count'].iat[i]
                y = db['comment_count'].iat[i + 1]
                db['commentCountSinceLast'].iat[i] = int(x - y)
            else:
                db['commentCountSinceLast'].iat[i] = int(db['comment_count'][i])
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['comment_count'].iat[i]
                y = db['comment_count'].iat[i + 1]
                db['commentCountSinceLast'].iat[i] = int(x - y)
            else:
                db['commentCountSinceLast'].iat[i] = int(db['comment_count'][i])


def likesBetweenSamples(db):
    dataBase.insert(loc=12, column='likesSinceLast', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['likes'].iat[i]
                y = db['likes'].iat[i + 1]
                db['likesSinceLast'].iat[i] = int(x - y)
            else:
                db['likesSinceLast'].iat[i] = int(db['likes'][i])
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['likes'].iat[i]
                y = db['likes'].iat[i + 1]
                db['likesSinceLast'].iat[i] = int(x - y)
            else:
                db['likesSinceLast'].iat[i] = int(db['likes'][i])


def dislikesBetweenSamples(db):
    dataBase.insert(loc=12, column='dislikesSinceLast', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['dislikes'].iat[i]
                y = db['dislikes'].iat[i + 1]
                db['dislikesSinceLast'].iat[i] = int(x - y)
            else:
                db['dislikesSinceLast'].iat[i] = int(db['dislikes'][i])
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['dislikes'].iat[i]
                y = db['dislikes'].iat[i + 1]
                db['dislikesSinceLast'].iat[i] = int(x - y)
            else:
                db['dislikesSinceLast'].iat[i] = int(db['dislikes'][i])


def daysSincePublised(db):
    dataBase.insert(loc=12, column='daysSincePub', value='x')
    for i in range(0, len(db)):
        x = db['Trending'].iat[i]
        y = db['PublishDate'].iat[i]
        db['daysSincePub'].iat[i] = int(calculateDays(x, y))


def commentCountPerDay(db):
    dataBase.insert(loc=12, column='commentCountPerDay', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['commentCountSinceLast'].iat[i]
                y = db['daysSinceLast'].iat[i]
                db['commentCountPerDay'].iat[i] = int(x / y)
            else:
                x = db['commentCountSinceLast'].iat[i]
                y = db['daysSincePub'].iat[i]
                if y == 0:
                    db['commentCountPerDay'].iat[i] = int(x)
                else:
                    db['commentCountPerDay'].iat[i] = int(x / y)
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['commentCountSinceLast'].iat[i]
                y = db['daysSinceLast'].iat[i]
                db['commentCountPerDay'].iat[i] = int(x / y)
            else:
                x = db['commentCountSinceLast'].iat[i]
                y = db['daysSincePub'].iat[i]
                if y == 0:
                    db['commentCountPerDay'].iat[i] = int(x)
                else:
                    db['commentCountPerDay'].iat[i] = int(x / y)


def likesPerDay(db):
    dataBase.insert(loc=12, column='likesPerDay', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['likesSinceLast'].iat[i]
                y = db['daysSinceLast'].iat[i]
                db['likesPerDay'].iat[i] = int(x / y)
            else:
                x = db['likesSinceLast'].iat[i]
                y = db['daysSincePub'].iat[i]
                if y == 0:
                    db['likesPerDay'].iat[i] = int(x)
                else:
                    db['likesPerDay'].iat[i] = int(x / y)
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['likesSinceLast'].iat[i]
                y = db['daysSinceLast'].iat[i]
                db['likesPerDay'].iat[i] = int(x / y)
            else:
                x = db['likesSinceLast'].iat[i]
                y = db['daysSincePub'].iat[i]
                if y == 0:
                    db['likesPerDay'].iat[i] = int(x)
                else:
                    db['likesPerDay'].iat[i] = int(x / y)


def dislikesPerDay(db):
    dataBase.insert(loc=12, column='dislikesPerDay', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['dislikesSinceLast'].iat[i]
                y = db['daysSinceLast'].iat[i]
                db['dislikesPerDay'].iat[i] = int(x / y)
            else:
                x = db['dislikesSinceLast'].iat[i]
                y = db['daysSincePub'].iat[i]
                if y == 0:
                    db['dislikesPerDay'].iat[i] = int(x)
                else:
                    db['dislikesPerDay'].iat[i] = int(x / y)
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['dislikesSinceLast'].iat[i]
                y = db['daysSinceLast'].iat[i]
                db['dislikesPerDay'].iat[i] = int(x / y)
            else:
                x = db['dislikesSinceLast'].iat[i]
                y = db['daysSincePub'].iat[i]
                if y == 0:
                    db['dislikesPerDay'].iat[i] = int(x)
                else:
                    db['dislikesPerDay'].iat[i] = int(x / y)


def commentCountPaceBetweenSamples(db):
    dataBase.insert(loc=12, column='commentPace', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['commentCountPerDay'].iat[i]
                y = db['commentCountPerDay'].iat[i + 1]
                if x > y:
                    db['commentPace'].iat[i] = 1
                else:
                    db['commentPace'].iat[i] = 0
            else:
                db['commentPace'].iat[i] = 1
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['commentCountPerDay'].iat[i]
                y = db['commentCountPerDay'].iat[i + 1]
                if x > y:
                    db['commentPace'].iat[i] = 1
                else:
                    db['commentPace'].iat[i] = 0
            else:
                db['commentPace'].iat[i] = 1


def dislikesPaceBetweenSamples(db):
    dataBase.insert(loc=12, column='dislikesPace', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['dislikesPerDay'].iat[i]
                y = db['dislikesPerDay'].iat[i + 1]
                if x > y:
                    db['dislikesPace'].iat[i] = 1
                else:
                    db['dislikesPace'].iat[i] = 0
            else:
                db['dislikesPace'].iat[i] = 1
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['dislikesPerDay'].iat[i]
                y = db['dislikesPerDay'].iat[i + 1]
                if x > y:
                    db['dislikesPace'].iat[i] = 1
                else:
                    db['dislikesPace'].iat[i] = 0
            else:
                db['dislikesPace'].iat[i] = 1


def likesPaceBetweenSamples(db):
    dataBase.insert(loc=12, column='likesPace', value='x')
    st = db['title'][0]
    for i in range(0, len(db)):
        if st == db['title'][i]:
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['likesPerDay'].iat[i]
                y = db['likesPerDay'].iat[i + 1]
                if x > y:
                    db['likesPace'].iat[i] = 1
                else:
                    db['likesPace'].iat[i] = 0
            else:
                db['likesPace'].iat[i] = 1
        else:
            st = db['title'][i]
            counter = db['SerialNum'][i]
            if counter > 1:
                x = db['likesPerDay'].iat[i]
                y = db['likesPerDay'].iat[i + 1]
                if x > y:
                    db['likesPace'].iat[i] = 1
                else:
                    db['likesPace'].iat[i] = 0
            else:
                db['likesPace'].iat[i] = 1


def summeryToExtraction(dataBase):
    DBE = dataBase.drop(
        ['title', 'publishedAt', 'channelId', 'channelTitle', 'categoryId', 'trending_date', 'tags', 'view_count',
         'likes', 'dislikes', 'comment_count', 'description', 'tagsArray', 'URLsArray', 'Trending', 'PublishDate',
         'PublishHour', 'daysSinceLast', 'SerialNum'], 1)
    DBE1 = DBE.drop(['likesPace', 'dislikesPace', 'commentPace', 'dislikesPerDay', 'likesPerDay', 'commentCountPerDay',
                     'daysSincePub', 'text'], 1)
    DBE2 = DBE.drop(['likesPace', 'dislikesPace', 'commentPace', 'dislikesPerDay', 'likesPerDay', 'commentCountPerDay',
                     'dislikesSinceLast', 'likesSinceLast', 'commentCountSinceLast', 'isVEVO'], 1)
    DBE3 = DBE.drop(
        ['likesPace', 'dislikesPace', 'commentPace', 'dislikesSinceLast', 'likesSinceLast', 'commentCountSinceLast',
         'isVEVO', 'daysSincePub', 'text'], 1)
    DBE4 = DBE.drop(['dislikesPerDay', 'likesPerDay', 'commentCountPerDay', 'dislikesSinceLast', 'likesSinceLast',
                     'commentCountSinceLast', 'isVEVO', 'daysSincePub', 'text'], 1)
    DBE5 = DBE.drop(['likesPace', 'dislikesPace', 'commentPace', 'dislikesPerDay', 'likesPerDay', 'commentCountPerDay',
                     'dislikesSinceLast', 'likesSinceLast', 'commentCountSinceLast', 'daysSincePub'], 1)
    print(DBE1.describe())
    print(DBE2.describe())
    print(DBE3.describe())
    print(DBE4.describe())


def featureExtraction(dataBase):
    # fixed number
    concatToString(dataBase)
    isVEVO(dataBase)
    # rep

    # MTSD
    sampleSerialNum(dataBase)
    daysBetweenSamples(dataBase)
    commentCountBetweenSamples(dataBase)
    likesBetweenSamples(dataBase)
    dislikesBetweenSamples(dataBase)
    daysSincePublised(dataBase)
    commentCountPerDay(dataBase)
    likesPerDay(dataBase)
    dislikesPerDay(dataBase)

    # Between samples
    commentCountPaceBetweenSamples(dataBase)
    dislikesPaceBetweenSamples(dataBase)
    likesPaceBetweenSamples(dataBase)

    summeryToExtraction(dataBase)


def fillArray(string):
    array = []
    temp = 0
    for i in range(0, len(string)):
        if string[i] == '|':
            array.append(string[temp:i])
            temp = i + 1
    array.append(string[temp:len(string)])
    return array


def putIntoArry(db):
    db.insert(loc=12, column='tagsArray', value='x')
    for i in range(0, len(db)):
        x = fillArray(db['tags'][i])
        db['tagsArray'].iat[i] = x


def fillURL(string):
    array = []
    temp = 0
    bool = False
    for i in range(0, len(string) - 4):
        if string[i:i + 4] == 'http':
            for j in range(i, len(string)):
                if string[j] == ' ' or j == len(string) - 1:
                    st2 = string[i:j + 1]
                    if st2[0:4] == 'http':
                        array.append(st2)
                        bool = True
                        break
    if bool == False:
        array.append('None')
    return array


def seperateURLS(db):
    db.insert(loc=12, column='URLsArray', value='x')
    for i in range(0, len(db)):
        x = fillURL(db['description'][i])
        db['URLsArray'].iat[i] = x


def removeHour(s):
    string = ''
    for i in range(0, len(s)):
        if s[i] == 'T':
            string = s[0:i]
            break
    return string


def removeTrendingHour(db):
    db.insert(loc=12, column='Trending', value='x')
    for i in range(0, len(db)):
        x = removeHour(db['trending_date'][i])
        db['Trending'].iat[i] = x


def removePublishedHour(db):
    db.insert(loc=12, column='PublishDate', value='x')
    for i in range(0, len(db)):
        x = removeHour(db['publishedAt'][i])
        db['PublishDate'].iat[i] = x


def seperateHour(dateTime):
    hour = 0
    for i in range(0, len(dateTime)):
        if dateTime[i] == 'T':
            hour = int(dateTime[i + 1:i + 3])
            break
    return hour


def seperateHourFromDate(db):
    db.insert(loc=12, column='PublishHour', value=0)
    for i in range(0, len(db)):
        x = seperateHour(db['publishedAt'][i])
        db['PublishHour'].iat[i] = x


def seperatePublishedDataTime(db):
    removePublishedHour(db)
    seperateHourFromDate(db)


def segmentation(db):
    putIntoArry(db)
    seperateURLS(db)
    removeTrendingHour(db)
    seperatePublishedDataTime(db)
    printPublishHour_dist(db)


def printPublishHour_dist(db):
    plt.boxplot(dataBase['PublishHour'])
    plt.show()


def missingLikes(dataBase):
    for i in range(0, len(dataBase)):
        if pd.isnull(dataBase['likes'][i]):
            title = dataBase['title'].iat[i]
            trending = dataBase['trending_date'].iat[i]
            dataErased['s_title'].iat[i] = dataBase['title'].iat[i]
            dataErased['s_trending_date'].iat[i] = dataBase['trending_date'].iat[i]
            print("drop sample of ", title, "from trending date ", trending)
    DB = dataBase
    X = DB.dropna(subset=['likes'])
    X.reset_index(drop=True, inplace=True)
    dataBase = X
    print(dataBase.describe())


def MissingDescriptions(dataBase):
    for i in range(0, len(dataBase)):
        if pd.isnull(dataBase['description'][i]):
            dataBase['description'].iat[i] = '[None]'
            title = dataBase['title'].iat[i]
            trending = dataBase['trending_date'].iat[i]
            dataFixed['s_title'].iat[i] = dataBase['title'].iat[i]
            dataFixed['s_trending_date'].iat[i] = dataBase['trending_date'].iat[i]
            print("updated description sample of ", title, "from trending date ", trending, "to [None]")


def completeDescriptions(dataBase):
    SqlQuery1 = cur.execute("""
                SELECT title, Count(*) as num, result, max(description) as val 
                FROM TrainingSet as T JOIN (SELECT title as t1, Count(description) as result 
                                            FROM TrainingSet
                                            WHERE description IS NOT NULL
                                            GROUP BY title
                                            ) as J ON T.title = J.t1

                GROUP BY title
            """)
    res1 = pd.DataFrame(SqlQuery1, columns=['title', 'num', 'result', 'val'])
    for i in range(0, len(res1)):
        if res1['result'][i] < res1['num'][i]:
            temp = res1['title'][i]
            v = res1['val'][i]
            for j in range(0, len(dataBase)):
                if dataBase['title'][j].strip() == temp:
                    if pd.isnull(dataBase['description'][j]):
                        dataBase['description'].iat[j] = v
                        dataFixed['s_title'].iat[j] = dataBase['title'].iat[j]
                        dataFixed['s_trending_date'].iat[j] = dataBase['trending_date'].iat[j]
                        dataFixed['s_description'].iat[j] = dataBase['description'].iat[j]
                        print(v, " description value was added to the data collection ", "to the video ", temp,
                              "of trending date ", dataBase['trending_date'].iat[j])


def missingDescription(db):
    completeDescriptions(db)
    MissingDescriptions(db)


def dealMissingData(db):
    missingCategoryID(db)
    missingLikes(db)
    missingDescription(db)
    print("data fixed :")
    print(dataFixed.describe())
    print("data erased :")
    print(dataErased.describe())


def missingCategoryID(dataBase):
    SqlQuery = cur.execute("""
                SELECT title, Count(*) as num, result , max(categoryId) as val
                FROM TrainingSet as T JOIN (SELECT title as t1, Count(categoryId) as result 
                                            FROM TrainingSet
                                            WHERE categoryId >0
                                            GROUP BY title
                                            ) as J ON T.title = J.t1
                GROUP BY title
            """)
    res = pd.DataFrame(SqlQuery, columns=['title', 'num', 'result', 'val'])
    for i in range(0, len(res)):
        if res['result'][i] < res['num'][i]:
            temp = res['title'][i]
            v = res['val'][i]
            for j in range(0, len(dataBase)):
                if dataBase['title'][j].strip() == temp:
                    if pd.isnull(dataBase['categoryId'][j]):
                        dataBase['categoryId'].iat[j] = v
                        dataFixed['s_title'].iat[j] = dataBase['title'].iat[j]
                        dataFixed['s_trending_date'].iat[j] = dataBase['trending_date'].iat[j]
                        dataFixed['s_categoryId'].iat[j] = dataBase['categoryId'].iat[j]
                        print(v, " categoryId value was added to the data collection ", "to the video ", temp,
                              "of trending date ", dataBase['trending_date'].iat[j])


def insertInfo(db, i, s_title, s_publishedAt, s_channelId, s_channelTitle, s_categoryId, s_trending_date, s_tags,
               s_view_count, s_likes, s_dislikes, s_comment_count, s_description):
    db['title'][i] = s_title
    db['publishedAt'][i] = s_publishedAt
    db['channelId'][i] = s_channelId
    db['channelTitle'][i] = s_channelTitle
    db['categoryId'][i] = s_categoryId
    db['trending_date'][i] = s_trending_date
    db['tags'][i] = s_tags
    db['view_count'][i] = s_view_count
    db['likes'][i] = s_likes
    db['dislikes'][i] = s_dislikes
    db['comment_count'][i] = s_comment_count
    db['description'][i] = s_description


def printDescribe(db):
    print(db.describe())


def dropTable():
    _conn.executescript("""
            DROP TABLE TrainingSet

        """)


def createTable():
    _conn.executescript("""
            CREATE TABLE TrainingSet (
                title varchar(2505),
                publishedAt varchar(2505),
                channelId varchar(2505),
                channelTitle varchar(2505),
                categoryId INTEGER,
                trending_date varchar(2505), 
                tags varchar(2505),
                view_count INTEGER,
                likes INTEGER,
                dislikes INTEGER,
                comment_count INTEGER,
                description varchar(2505),
                PRIMARY KEY(title, trending_date) 
            );
        """)


def insert_sample(s_title, s_publishedAt, s_channelId, s_channelTitle, s_categoryId, s_trending_date, s_tags,
                  s_view_count, s_likes, s_dislikes, s_comment_count, s_description):
    _conn.execute("""
        INSERT INTO TrainingSet (title, publishedAt, channelId, channelTitle, categoryId, trending_date, tags, view_count, likes, dislikes, comment_count, description) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                  [s_title, s_publishedAt, s_channelId, s_channelTitle, s_categoryId, s_trending_date, s_tags,
                   s_view_count, s_likes, s_dislikes, s_comment_count, s_description])


def pasteData(db):
    for i in range(0, len(db)):
        s_title = db['title'][i].strip()
        s_publishedAt = db['publishedAt'][i].strip()
        s_channelId = db['channelId'][i].strip()
        s_channelTitle = db['channelTitle'][i].strip()
        s_categoryId = db['categoryId'][i]
        s_trending_date = db['trending_date'][i].strip()
        s_tags = db['tags'][i].strip()
        s_view_count = db['view_count'][i]
        s_likes = db['likes'][i]
        s_dislikes = db['dislikes'][i]
        s_comment_count = db['comment_count'][i]
        if pd.isnull(db['description'][i]):
            s_description = db['description'][i]
        else:
            s_description = db['description'][i].strip()
        insert_sample(s_title, s_publishedAt, s_channelId, s_channelTitle, s_categoryId, s_trending_date, s_tags,
                      s_view_count, s_likes, s_dislikes, s_comment_count, s_description)


def printCategoryId_dist(db):
    plt.plot(db['categoryId'], linewidth=0.3)
    plt.title("categoryId_dist", fontsize=20)
    plt.show()


def printView_count_dist(db):
    plt.plot(db['view_count'], linewidth=0.3)
    plt.title("view_count_dist", fontsize=20)
    plt.show()


def printLikes_dist(db):
    plt.plot(db['likes'], linewidth=0.3)
    plt.title("likes_dist", fontsize=20)
    plt.show()


def printDislikes_dist(db):
    plt.plot(db['dislikes'], linewidth=0.3)
    plt.title("dislikes_dist", fontsize=20)
    plt.show()


def printComment_count_dist(db):
    plt.plot(db['comment_count'], linewidth=0.3)
    plt.title("comment_count_dist", fontsize=20)
    plt.show()


def printCorr(db):
    DataCorr = db.corr(method='pearson')
    print(DataCorr)


def likes_VS_dislikes(db):
    plt.scatter(x=db['likes'], y=db['dislikes'])
    plt.xlabel('likes', fontsize=12)
    plt.ylabel('dislikes', fontsize=12)
    plt.title("likes_VS_dislikes", fontsize=20)
    plt.show()


def likes_VS_comment_count(db):
    plt.scatter(x=db['likes'], y=db['comment_count'])
    plt.xlabel('likes', fontsize=12)
    plt.ylabel('comment_count', fontsize=12)
    plt.title("likes_VS_comment_count", fontsize=20)
    plt.show()


def dislikes_VS_comment_count(db):
    plt.scatter(x=db['dislikes'], y=db['comment_count'])
    plt.xlabel('dislikes', fontsize=12)
    plt.ylabel('comment_count', fontsize=12)
    plt.title("dislikes_VS_comment_count", fontsize=20)
    plt.show()






