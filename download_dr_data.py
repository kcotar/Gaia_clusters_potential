import os
import urllib2
from bs4 import BeautifulSoup

# all data
source_url = 'http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'
save_dir = '/home/klemen/data4_mount/Gaia_DR2'

# data with rv
# source_url = 'http://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source_with_rv/csv/'
# save_dir = '/home/klemen/data4_mount/Gaia_DR2_RV'

os.system('mkdir '+save_dir)
os.chdir(save_dir)

dl_page = urllib2.urlopen(source_url)
dl_page_html = BeautifulSoup(dl_page, 'html.parser')

list_dl_url = dl_page_html.findAll('a')

for dl_url in list_dl_url:
    href_url = dl_url.get('href')
    if 'csv.gz' in href_url:
        print 'Downloading: '+href_url.split('/')[-1]
        os.system('wget '+source_url+href_url)
        