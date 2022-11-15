import os
import pandas as pd

def getfilename(filePath):

    name = os.listdir(filePath)
    df = pd.DataFrame(name, columns=['filename'])
    df['label'] = df['filename'].apply(lambda x:x[0:3])
    print(df.head())
    return df

if __name__ == '__main__':
    filePath = 'E:\Documents\Matlab_work\DataBase\IITD Palmprint V1\Segmented\Left'
    labelfile = getfilename(filePath)
    labelfile.to_csv('E:\Documents\Matlab_work\DataBase\IITD Palmprint V1\Segmented'+'/train.csv', index=False)

