# coding: utf-8
"""
this code is used to download an extract CIFAR-10 data set.
"""
import sys
import os
import urllib
import tarfile

def _print_download_progress(count, block_size, total_size):
    """
    function used for printing the download progress.
    Used as a call-back function in maybe_download_and_extract().
    """
    
    # percentage completion.
    pct_complete = float(count*block_size)/total_size
    
    # Status message. 
    msg = "\r- Download progress: {0:.1%}".format(pct_complete) #'\r':当一行打印结束后,再从该行开始位置打印
    
    # Print
    sys.stdout.write(msg) # 相当于print(但最后不会添加换行符)
    sys.stdout.flush() # 输出缓冲,以便实时显示进度

def maybe_download_and_extract(url, download_dir):
    """
    Download and extract the data if it doesn't already exist.
    Assume the url is a tar-ball file
    
    Args:
        url: Internet URL for the tar-file to download.    
        download_dir: Directory where the download file is saved.
    """
    
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir,filename)
    
    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.mkdir(download_dir) 
        """
        # for python2
        file_path, _ = urllib.urlretrieve(url=url, 
                                          filename=file_path, 
                                          reporthook=_print_download_progress)
        """
        # for python3
        file_path, _ = urllib.request.urlretrieve(url=url, 
                                          filename=file_path, 
                                          reporthook=_print_download_progress)
        print("download finished.") 
    else:
        print("unpacking...")
        tarfile.open(name=file_path, mode='r:gz').extractall(download_dir)
        print("Data has apparently already been download and unpacked!")     