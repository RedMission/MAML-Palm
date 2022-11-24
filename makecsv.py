import os
import pandas as pd

def get_IITD_filename(filePath):
    name = os.listdir(filePath)
    df = pd.DataFrame(name, columns=['filename'])
    df['label'] = df['filename'].apply(lambda x:x[0:3])
    print(df.head())
    return df

def get_MPDv2_filename(filePath):
    name = os.listdir(filePath)
    df = pd.DataFrame(name, columns=['filename'])
    df['label'] = df['filename'].apply(lambda x:(x[0:4]+x[8]))
    print(df.head())
    return df

def get_PolyU_filename(filePath):
    name = os.listdir(filePath)
    df = pd.DataFrame(name, columns=['filename'])
    df['label'] = df['filename'].apply(lambda x:(x[6:8]))
    print(df.head())
    return df

if __name__ == '__main__':
    # filePath = 'E:\Documents\Matlab_work\DataBase\IITD Palmprint V1\Segmented\Left'
    # labelfile = get_IITD_filename(filePath)
    # labelfile.to_csv('E:\Documents\Matlab_work\DataBase\IITD Palmprint V1\Segmented'+'/train.csv', index=False)

    # filePath = 'E:\Documents\Matlab_work\DataBase\MPDv2\generations/2-ROI verification\ROI'
    # labelfile = get_MPDv2_filename(filePath)
    # labelfile.to_csv('E:\Documents\Matlab_work\DataBase\MPDv2\generations/2-ROI verification'+'/train.csv', index=False)

    filePath = 'E:\Documents\Matlab_work\DataBase\PolyU\PolyUROI'
    labelfile = get_PolyU_filename(filePath)
    labelfile.to_csv('E:\Documents\Matlab_work\DataBase\PolyU' + '/roi.csv', index=False)


