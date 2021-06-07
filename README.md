```python
from google.colab import drive 
drive.mount("gdrive")
```

    Drive already mounted at gdrive; to attempt to forcibly remount, call drive.mount("gdrive", force_remount=True).
    


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gensim.downloader
import nltk
import string 
import seaborn as sn
```


```python
df = pd.read_csv("labeled_qnas.csv",index_col=0)
```

Lets check what we have imported.


```python
df.head(-1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GroupId</th>
      <th>SubGroupId</th>
      <th>Categories</th>
      <th>Q&amp;A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>Web Formları</td>
      <td>Web sitesi üzerinden yaptığım başvuruyu nasıl ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>Web Formları</td>
      <td>Kredi kartı başvurumu nereden yapabilirim?\nTü...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>Web Formları</td>
      <td>Web sitesi üzerinden yabancı uyruklu müşterile...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>Web Formları</td>
      <td>Web sitesinden sadece yeni kredi kartı için mi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>Web Formları</td>
      <td>Kredi kartı şifremi nasıl belirleyebilirim?\nB...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>361</th>
      <td>3</td>
      <td>3</td>
      <td>Bireysel İhtiyaç</td>
      <td>Başvuru için gerekli belgeler (aslı ve birer f...</td>
    </tr>
    <tr>
      <th>362</th>
      <td>4</td>
      <td>0</td>
      <td>SWIFT</td>
      <td>Swift kodu nedir?\nBankamız swift kodu YAPITRI...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>4</td>
      <td>1</td>
      <td>Düzenli Ödeme</td>
      <td>Üniversite ödemelerimi nasıl gerçekleştirebili...</td>
    </tr>
    <tr>
      <th>364</th>
      <td>4</td>
      <td>1</td>
      <td>Düzenli Ödeme</td>
      <td>Sıkça yaptığım bir ödememin hesabımdan önceden...</td>
    </tr>
    <tr>
      <th>365</th>
      <td>4</td>
      <td>1</td>
      <td>Düzenli Ödeme</td>
      <td>Hesabımdan, bir yakınımın hesabına belli dönem...</td>
    </tr>
  </tbody>
</table>
<p>366 rows × 4 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 367 entries, 0 to 366
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   GroupId     367 non-null    int64 
     1   SubGroupId  367 non-null    int64 
     2   Categories  367 non-null    object
     3   Q&A         367 non-null    object
    dtypes: int64(2), object(2)
    memory usage: 14.3+ KB
    

The dataset seems clean.


```python
df.GroupId.unique()
```




    array([0, 1, 2, 3, 4])




```python
df.SubGroupId.unique()
```




    array([0, 1, 2, 3, 4])




```python
df.Categories.unique()
```




    array(['Web Formları', 'Mobil Bankacılık', 'İnternet Şubesi',
           'Güvenlik Ürünleri', 'ATM', 'Üye İşyeri', 'Kredi Kartları',
           'Banka Kartları', 'Vadesiz Hesap', 'Genel', 'Vadeli Hesap',
           'Altın Hesabı', 'Konut', 'Esnek Hesap', 'Taşıt',
           'Bireysel İhtiyaç', 'SWIFT', 'Düzenli Ödeme'], dtype=object)



# Pre-processing


```python
df.drop(columns=["GroupId","SubGroupId"],inplace=True)
```

Actually, we dont need to work with Group and SubGroup columsn.


```python
df["Q&A"]
```




    0      Web sitesi üzerinden yaptığım başvuruyu nasıl ...
    1      Kredi kartı başvurumu nereden yapabilirim?\nTü...
    2      Web sitesi üzerinden yabancı uyruklu müşterile...
    3      Web sitesinden sadece yeni kredi kartı için mi...
    4      Kredi kartı şifremi nasıl belirleyebilirim?\nB...
                                 ...                        
    362    Swift kodu nedir?\nBankamız swift kodu YAPITRI...
    363    Üniversite ödemelerimi nasıl gerçekleştirebili...
    364    Sıkça yaptığım bir ödememin hesabımdan önceden...
    365    Hesabımdan, bir yakınımın hesabına belli dönem...
    366    Her ay yaptığım kira ödememin hesabımdan önced...
    Name: Q&A, Length: 367, dtype: object



We can deal with certain issues like endline and etc.


```python
for index, row in df.iterrows():
    if(index==5):
      break
    print(row[1])
    print("\n")
```

    Web sitesi üzerinden yaptığım başvuruyu nasıl takip edebilirim?
    Kredi kartı başvurunuzu web sitesinden Kredi Kartı Başvuru Sorgulama adımından hemen sorgulayabilirsniz. Başvurunuz varsa hemen sorgulamak için tıklayınız.
    
    
    Kredi kartı başvurumu nereden yapabilirim?
    Türkiye'nin en geniş kart portföyü olan World'e sahip olmak için şubeye gitmeden internetten, SMS ile veya Yapı Kredi Telefon Bankacılığı üzerinden başvuru yapabilirsiniz. Kredi kartına hemen başvurmak için tıklayınız.
    Diğer başvuru kanalları hakkında detaylı bilgi için tıklayınız.
    
    
    Web sitesi üzerinden yabancı uyruklu müşteriler şifre belirleyebilir mi?
    Evet, yabancı uyruklu müşterilerimiz bankamızda kayıtlı Yabancı Kimlik Numaralarını kullanarak şifre belirleyebilirler.
    
    
    Web sitesinden sadece yeni kredi kartı için mi şifre belirleyebilirim?
    Hayır, sadece yeni kartınız için değil, şifresini unuttuğunuz ya da değiştirmek istediğiniz kartınız için de şifre belirleme işlemi yapabilirsiniz.
    
    
    Kredi kartı şifremi nasıl belirleyebilirim?
    Bireysel veya Ticari kredi kartlarınız için kredi kart şifrenizi Yapı Kredi Web Sitesi, Bireysel İnternet Şubesi, Mobil Şube, Nuvo İnternet Bankacılığı ve SMS aracılığı ile belirleyebilirsiniz. Detaylı bilgi için tıklayınız.
    Belirlediğiniz şifre ile www.yapikredi.com.tr üzerinde yer alan "Kart İşlemlerim"e üye olabilirsiniz. Hesap özeti görüntüleme, kart borcu ödeme gibi kart işlemleriniz için Kart İşlemlerim adımından internet şubesi ve mobil şubeye giriş yapabilirsiniz.
    
    
    

## Punctuation Removal


```python
df["Q&A"]=df["Q&A"].str.replace('[{}]'.format(string.punctuation), '')
df.head(-1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categories</th>
      <th>Q&amp;A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Web Formları</td>
      <td>Web sitesi üzerinden yaptığım başvuruyu nasıl ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Web Formları</td>
      <td>Kredi kartı başvurumu nereden yapabilirim\nTür...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Web Formları</td>
      <td>Web sitesi üzerinden yabancı uyruklu müşterile...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Web Formları</td>
      <td>Web sitesinden sadece yeni kredi kartı için mi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Web Formları</td>
      <td>Kredi kartı şifremi nasıl belirleyebilirim\nBi...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>361</th>
      <td>Bireysel İhtiyaç</td>
      <td>Başvuru için gerekli belgeler aslı ve birer fo...</td>
    </tr>
    <tr>
      <th>362</th>
      <td>SWIFT</td>
      <td>Swift kodu nedir\nBankamız swift kodu YAPITRIS...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Düzenli Ödeme</td>
      <td>Üniversite ödemelerimi nasıl gerçekleştirebili...</td>
    </tr>
    <tr>
      <th>364</th>
      <td>Düzenli Ödeme</td>
      <td>Sıkça yaptığım bir ödememin hesabımdan önceden...</td>
    </tr>
    <tr>
      <th>365</th>
      <td>Düzenli Ödeme</td>
      <td>Hesabımdan bir yakınımın hesabına belli döneml...</td>
    </tr>
  </tbody>
</table>
<p>366 rows × 2 columns</p>
</div>



## Lower Casing


```python
df["Q&A"]=df["Q&A"].str.lower()
df.head(-1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categories</th>
      <th>Q&amp;A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Web Formları</td>
      <td>web sitesi üzerinden yaptığım başvuruyu nasıl ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Web Formları</td>
      <td>kredi kartı başvurumu nereden yapabilirim\ntür...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Web Formları</td>
      <td>web sitesi üzerinden yabancı uyruklu müşterile...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Web Formları</td>
      <td>web sitesinden sadece yeni kredi kartı için mi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Web Formları</td>
      <td>kredi kartı şifremi nasıl belirleyebilirim\nbi...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>361</th>
      <td>Bireysel İhtiyaç</td>
      <td>başvuru için gerekli belgeler aslı ve birer fo...</td>
    </tr>
    <tr>
      <th>362</th>
      <td>SWIFT</td>
      <td>swift kodu nedir\nbankamız swift kodu yapitris...</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Düzenli Ödeme</td>
      <td>üniversite ödemelerimi nasıl gerçekleştirebili...</td>
    </tr>
    <tr>
      <th>364</th>
      <td>Düzenli Ödeme</td>
      <td>sıkça yaptığım bir ödememin hesabımdan önceden...</td>
    </tr>
    <tr>
      <th>365</th>
      <td>Düzenli Ödeme</td>
      <td>hesabımdan bir yakınımın hesabına belli döneml...</td>
    </tr>
  </tbody>
</table>
<p>366 rows × 2 columns</p>
</div>



## Stopwords Removal


```python
stopwords = []
with open("/content/gdrive/MyDrive/CS-445/stopwords-extended") as s_file:
  stopwords = s_file.readlines()
```


```python
stopwords = [stopword[:-1] for stopword in stopwords]
```


```python
stopwords
```




    ['acaba',
     'acep',
     'adamakıllı',
     'adeta',
     'ait',
     'altmýþ',
     'altmış',
     'altý',
     'altı',
     'ama',
     'amma',
     'anca',
     'ancak',
     'arada',
     'artýk',
     'aslında',
     'aynen',
     'ayrıca',
     'az',
     'açıkça',
     'açıkçası',
     'bana',
     'bari',
     'bazen',
     'bazý',
     'bazı',
     'başkası',
     'baţka',
     'belki',
     'ben',
     'benden',
     'beni',
     'benim',
     'beri',
     'beriki',
     'beþ',
     'beş',
     'beţ',
     'bilcümle',
     'bile',
     'bin',
     'binaen',
     'binaenaleyh',
     'bir',
     'biraz',
     'birazdan',
     'birbiri',
     'birden',
     'birdenbire',
     'biri',
     'birice',
     'birileri',
     'birisi',
     'birkaç',
     'birkaçı',
     'birkez',
     'birlikte',
     'birçok',
     'birçoğu',
     'birþey',
     'birþeyi',
     'birşey',
     'birşeyi',
     'birţey',
     'bitevi',
     'biteviye',
     'bittabi',
     'biz',
     'bizatihi',
     'bizce',
     'bizcileyin',
     'bizden',
     'bize',
     'bizi',
     'bizim',
     'bizimki',
     'bizzat',
     'boşuna',
     'bu',
     'buna',
     'bunda',
     'bundan',
     'bunlar',
     'bunları',
     'bunların',
     'bunu',
     'bunun',
     'buracıkta',
     'burada',
     'buradan',
     'burası',
     'böyle',
     'böylece',
     'böylecene',
     'böylelikle',
     'böylemesine',
     'böylesine',
     'büsbütün',
     'bütün',
     'cuk',
     'cümlesi',
     'da',
     'daha',
     'dahi',
     'dahil',
     'dahilen',
     'daima',
     'dair',
     'dayanarak',
     'de',
     'defa',
     'dek',
     'demin',
     'demincek',
     'deminden',
     'denli',
     'derakap',
     'derhal',
     'derken',
     'deđil',
     'değil',
     'değin',
     'diye',
     'diđer',
     'diğer',
     'diğeri',
     'doksan',
     'dokuz',
     'dolayı',
     'dolayısıyla',
     'doğru',
     'dört',
     'edecek',
     'eden',
     'ederek',
     'edilecek',
     'ediliyor',
     'edilmesi',
     'ediyor',
     'elbet',
     'elbette',
     'elli',
     'emme',
     'en',
     'enikonu',
     'epey',
     'epeyce',
     'epeyi',
     'esasen',
     'esnasında',
     'etmesi',
     'etraflı',
     'etraflıca',
     'etti',
     'ettiği',
     'ettiğini',
     'evleviyetle',
     'evvel',
     'evvela',
     'evvelce',
     'evvelden',
     'evvelemirde',
     'evveli',
     'eđer',
     'eğer',
     'fakat',
     'filanca',
     'gah',
     'gayet',
     'gayetle',
     'gayri',
     'gayrı',
     'gelgelelim',
     'gene',
     'gerek',
     'gerçi',
     'geçende',
     'geçenlerde',
     'gibi',
     'gibilerden',
     'gibisinden',
     'gine',
     'göre',
     'gırla',
     'hakeza',
     'halbuki',
     'halen',
     'halihazırda',
     'haliyle',
     'handiyse',
     'hangi',
     'hangisi',
     'hani',
     'hariç',
     'hasebiyle',
     'hasılı',
     'hatta',
     'hele',
     'hem',
     'henüz',
     'hep',
     'hepsi',
     'her',
     'herhangi',
     'herkes',
     'herkesin',
     'hiç',
     'hiçbir',
     'hiçbiri',
     'hoş',
     'hulasaten',
     'iken',
     'iki',
     'ila',
     'ile',
     'ilen',
     'ilgili',
     'ilk',
     'illa',
     'illaki',
     'imdi',
     'indinde',
     'inen',
     'insermi',
     'ise',
     'ister',
     'itibaren',
     'itibariyle',
     'itibarıyla',
     'iyi',
     'iyice',
     'iyicene',
     'için',
     'iş',
     'işte',
     'iţte',
     'kadar',
     'kaffesi',
     'kah',
     'kala',
     'kanýmca',
     'karşın',
     'katrilyon',
     'kaynak',
     'kaçı',
     'kelli',
     'kendi',
     'kendilerine',
     'kendini',
     'kendisi',
     'kendisine',
     'kendisini',
     'kere',
     'kez',
     'keza',
     'kezalik',
     'keşke',
     'keţke',
     'ki',
     'kim',
     'kimden',
     'kime',
     'kimi',
     'kimisi',
     'kimse',
     'kimsecik',
     'kimsecikler',
     'külliyen',
     'kýrk',
     'kýsaca',
     'kırk',
     'kısaca',
     'lakin',
     'leh',
     'lütfen',
     'maada',
     'madem',
     'mademki',
     'mamafih',
     'mebni',
     'međer',
     'meğer',
     'meğerki',
     'meğerse',
     'milyar',
     'milyon',
     'mu',
     'mü',
     'mý',
     'mı',
     'nasýl',
     'nasıl',
     'nasılsa',
     'nazaran',
     'naşi',
     'ne',
     'neden',
     'nedeniyle',
     'nedenle',
     'nedense',
     'nerde',
     'nerden',
     'nerdeyse',
     'nere',
     'nerede',
     'nereden',
     'neredeyse',
     'neresi',
     'nereye',
     'netekim',
     'neye',
     'neyi',
     'neyse',
     'nice',
     'nihayet',
     'nihayetinde',
     'nitekim',
     'niye',
     'niçin',
     'o',
     'olan',
     'olarak',
     'oldu',
     'olduklarını',
     'oldukça',
     'olduğu',
     'olduğunu',
     'olmadı',
     'olmadığı',
     'olmak',
     'olması',
     'olmayan',
     'olmaz',
     'olsa',
     'olsun',
     'olup',
     'olur',
     'olursa',
     'oluyor',
     'on',
     'ona',
     'onca',
     'onculayın',
     'onda',
     'ondan',
     'onlar',
     'onlardan',
     'onlari',
     'onlarýn',
     'onları',
     'onların',
     'onu',
     'onun',
     'oracık',
     'oracıkta',
     'orada',
     'oradan',
     'oranca',
     'oranla',
     'oraya',
     'otuz',
     'oysa',
     'oysaki',
     'pek',
     'pekala',
     'peki',
     'pekçe',
     'peyderpey',
     'rağmen',
     'sadece',
     'sahi',
     'sahiden',
     'sana',
     'sanki',
     'sekiz',
     'seksen',
     'sen',
     'senden',
     'seni',
     'senin',
     'siz',
     'sizden',
     'sizi',
     'sizin',
     'sonra',
     'sonradan',
     'sonraları',
     'sonunda',
     'tabii',
     'tam',
     'tamam',
     'tamamen',
     'tamamıyla',
     'tarafından',
     'tek',
     'trilyon',
     'tüm',
     'var',
     'vardı',
     'vasıtasıyla',
     've',
     'velev',
     'velhasıl',
     'velhasılıkelam',
     'veya',
     'veyahut',
     'ya',
     'yahut',
     'yakinen',
     'yakında',
     'yakından',
     'yakınlarda',
     'yalnız',
     'yalnızca',
     'yani',
     'yapacak',
     'yapmak',
     'yaptı',
     'yaptıkları',
     'yaptığı',
     'yaptığını',
     'yapılan',
     'yapılması',
     'yapıyor',
     'yedi',
     'yeniden',
     'yenilerde',
     'yerine',
     'yetmiþ',
     'yetmiş',
     'yetmiţ',
     'yine',
     'yirmi',
     'yok',
     'yoksa',
     'yoluyla',
     'yüz',
     'yüzünden',
     'zarfında',
     'zaten',
     'zati',
     'zira',
     'çabuk',
     'çabukça',
     'çeşitli',
     'çok',
     'çokları',
     'çoklarınca',
     'çokluk',
     'çoklukla',
     'çokça',
     'çoğu',
     'çoğun',
     'çoğunca',
     'çoğunlukla',
     'çünkü',
     'öbür',
     'öbürkü',
     'öbürü',
     'önce',
     'önceden',
     'önceleri',
     'öncelikle',
     'öteki',
     'ötekisi',
     'öyle',
     'öylece',
     'öylelikle',
     'öylemesine',
     'öz',
     'üzere',
     'üç',
     'þey',
     'þeyden',
     'þeyi',
     'þeyler',
     'þu',
     'þuna',
     'þunda',
     'þundan',
     'þunu',
     'şayet',
     'şey',
     'şeyden',
     'şeyi',
     'şeyler',
     'şu',
     'şuna',
     'şuncacık',
     'şunda',
     'şundan',
     'şunlar',
     'şunları',
     'şunu',
     'şunun',
     'şura',
     'şuracık',
     'şuracıkta',
     'şurası',
     'şöyle',
     'ţayet',
     'ţimdi',
     'ţu',
     'ţöyl']



An extended collection of stopwords where 
t sometimes reduces the performance of the model in certain sentiment analysis tasks and etc.


```python
df["Q&A"]= df["Q&A"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
```


```python
for index, row in df.iterrows():
    if(index==5):
      break
    print(row[1])
    print("\n")
```

    web sitesi üzerinden yaptığım başvuruyu takip edebilirim kredi kartı başvurunuzu web sitesinden kredi kartı başvuru sorgulama adımından hemen sorgulayabilirsniz başvurunuz varsa hemen sorgulamak tıklayınız
    
    
    kredi kartı başvurumu yapabilirim türkiyenin geniş kart portföyü worlde sahip şubeye gitmeden internetten sms yapı kredi telefon bankacılığı üzerinden başvuru yapabilirsiniz kredi kartına hemen başvurmak tıklayınız başvuru kanalları hakkında detaylı bilgi tıklayınız
    
    
    web sitesi üzerinden yabancı uyruklu müşteriler şifre belirleyebilir mi evet yabancı uyruklu müşterilerimiz bankamızda kayıtlı yabancı kimlik numaralarını kullanarak şifre belirleyebilirler
    
    
    web sitesinden yeni kredi kartı mi şifre belirleyebilirim hayır yeni kartınız şifresini unuttuğunuz değiştirmek istediğiniz kartınız şifre belirleme işlemi yapabilirsiniz
    
    
    kredi kartı şifremi belirleyebilirim bireysel ticari kredi kartlarınız kredi kart şifrenizi yapı kredi web sitesi bireysel i̇nternet şubesi mobil şube nuvo i̇nternet bankacılığı sms aracılığı belirleyebilirsiniz detaylı bilgi tıklayınız belirlediğiniz şifre wwwyapikredicomtr üzerinde yer alan kart i̇şlemlerime üye olabilirsiniz hesap özeti görüntüleme kart borcu ödeme kart işlemleriniz kart i̇şlemlerim adımından internet şubesi mobil şubeye giriş yapabilirsiniz
    
    
    

## Number Removal

Not quite sure whether this is really necessary but since our data might contain percentages and etc we should check.


```python
df["Q&A"]=df["Q&A"].str.replace(r'\d+', '')
```

## Class Distribution 


```python
df["Categories"].hist(figsize=(30,10))

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f6f30ff03d0>




    
![png](FAQ_Classification_files/FAQ_Classification_31_1.png)
    



```python
df[df["Categories"]=="Altın Hesabı"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categories</th>
      <th>Q&amp;A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>300</th>
      <td>Altın Hesabı</td>
      <td>altın hesabının özellikleri nelerdir vadesiz a...</td>
    </tr>
    <tr>
      <th>301</th>
      <td>Altın Hesabı</td>
      <td>altın hesabının avantajları nelerdir çalınma r...</td>
    </tr>
    <tr>
      <th>302</th>
      <td>Altın Hesabı</td>
      <td>altın hesabı başvuru yapabilirim vadesiz altın...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df["Categories"]=="Vadeli Hesap"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categories</th>
      <th>Q&amp;A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>298</th>
      <td>Vadeli Hesap</td>
      <td>yurtdışında yaşıyorum bankanızda vadeli hesap ...</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Vadeli Hesap</td>
      <td>bankanızda vadeli hesap açtırmak istiyorum yap...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Categories'].value_counts()
```




    Kredi Kartları       98
    İnternet Şubesi      46
    Güvenlik Ürünleri    42
    Üye İşyeri           35
    ATM                  32
    Web Formları         24
    Bireysel İhtiyaç     20
    Konut                16
    Taşıt                16
    Mobil Bankacılık      8
    Esnek Hesap           7
    Banka Kartları        7
    Düzenli Ödeme         4
    Genel                 4
    Altın Hesabı          3
    Vadesiz Hesap         2
    Vadeli Hesap          2
    SWIFT                 1
    Name: Categories, dtype: int64




```python
df = df[df["Categories"]!="SWIFT"]
```


```python
df.replace("Altın Hesabı","Hesap",inplace=True)
df.replace("Vadesiz Hesap","Hesap",inplace=True)
df.replace("Vadeli Hesap","Hesap",inplace=True)
```


```python
df['Categories'].value_counts()
```




    Kredi Kartları       98
    İnternet Şubesi      46
    Güvenlik Ürünleri    42
    Üye İşyeri           35
    ATM                  32
    Web Formları         24
    Bireysel İhtiyaç     20
    Taşıt                16
    Konut                16
    Mobil Bankacılık      8
    Hesap                 7
    Esnek Hesap           7
    Banka Kartları        7
    Genel                 4
    Düzenli Ödeme         4
    Name: Categories, dtype: int64



# Word Embeddings

![image.png](data:image/png;base64,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qyP1wZfzte5QIsH38/7BsSECCCCQXwXSdGLbYk3/+C29OnSwBj0/TK+O/EBfzl+nQ3ncqKef3KJ5n76t14cN1fBRH2rqkj06c55TyXjTTT22Vt+MG6EXBz+n51+K0pjJS3XUR3Q/P7V96ad3a/k3E/R21Isa/OwgDR4WpTGfzdGqAwk5aIcvXL16qwu+QwABBBBAAAEfAta/NPOWcm6dIi5TuydnaF+OO1j42H+B/DpJG0ffo949e6pnz3567vtwfSKgQFZeWGSaIHxYVDOFRAABBBBAAAEEEEAAAQQQQAABBBC4eAQylBR7SHv3Reu0jw4DF09ZKQkCBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIITUowqwAAAqFJREFUIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBQjC59OKIVsIIIAAAggggAACCCCAAAIIIIAAAggggAACBV+AIHzBr0NKgAACCCCAAAIIIIAAAggggAACCCCAAAIIIJBPBf4fh4U4T8tBniYAAAAASUVORK5CYII=)


```python
# Importing libraries
from keras.datasets import imdb
from keras.models import Sequential,Model
from keras.layers import Dense,Input,Dropout
from keras.layers import Flatten,GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
from keras.layers.merge import concatenate
import gensim
from itertools import chain
from sklearn.model_selection import train_test_split
```


```python
w2v_1 = gensim.models.Word2Vec.load("/content/gdrive/MyDrive/CS-445/project02_word_vec.model")
w2v_2 = gensim.models.KeyedVectors.load_word2vec_format("/content/gdrive/MyDrive/CS-445/trmodel.model",binary=True)
pretrained_weight_1 = w2v_1.wv.vectors
pretrained_weight_2 = w2v_2.wv.vectors
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: DeprecationWarning: Call to deprecated `wv` (Attribute will be removed in 4.0.0, use self instead).
      after removing the cwd from sys.path.
    

There are 2 individual word embeddings in my possesion and i will use both of them.


```python
max = 0
for q in df["Q&A"]:
  tokenized = q.split(" ")
  max = len(tokenized) if len(tokenized) > max else max
```


```python
max
```




    314




```python
Target = pd.get_dummies(df["Categories"])
```


```python
x_train, x_test, y_train, y_test = train_test_split(df["Q&A"],Target,train_size=.8,random_state=1,stratify=Target,shuffle=True)
```


```python
len(y_test.value_counts())
```




    15




```python
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=314)
tokenizer.fit_on_texts(df["Q&A"])
```


```python
Xtrain = tokenizer.texts_to_matrix(x_train, mode='binary')
Xtest = tokenizer.texts_to_matrix(x_test, mode='binary')
```

# Naive Fully-Connected Network Model


```python
model = Sequential()
model.add(Input(314,))
model.add(Dense(512,activation='swish'))
model.add(Dense(256,activation='swish'))
model.add(Dense(128,activation='swish'))
model.add(Dense(64,activation='swish'))
model.add(Dense(32,activation='swish'))
model.add(Dense(15,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
```

    WARNING:tensorflow:Please add `keras.layers.InputLayer` instead of `keras.Input` to Sequential model. `keras.Input` is intended to be used by Functional model.
    Model: "sequential_9"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_65 (Dense)             (None, 512)               161280    
    _________________________________________________________________
    dense_66 (Dense)             (None, 256)               131328    
    _________________________________________________________________
    dense_67 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    dense_68 (Dense)             (None, 64)                8256      
    _________________________________________________________________
    dense_69 (Dense)             (None, 32)                2080      
    _________________________________________________________________
    dense_70 (Dense)             (None, 15)                495       
    =================================================================
    Total params: 336,335
    Trainable params: 336,335
    Non-trainable params: 0
    _________________________________________________________________
    


```python
x_train
```




    13     ödeme merkezi’nden kurumlara ödemeleri gerçekl...
    226    world gold kart nedir prestij kulüplerden gold...
    114    akıllı anahtarımı teslim alabilirim akıllı ana...
    146    kartsız işlem yapabilirim işlemleri yapı kredi...
    56     i̇nternet şubesinde eft işlemleri zaman yapıla...
                                 ...                        
    360    başvuru sonucumu zaman öğrenebilirim bireysel ...
    240    world sanal kart nedir world sanal kart’ınız k...
    152    yapı kredi pos’larından sanal pos dövizli işle...
    345    taksit ödeyeceğim tatil kredisiyle ücret faizl...
    227    world business kartlarına işveren mi çalışan b...
    Name: Q&A, Length: 292, dtype: object




```python
model.fit(Xtrain,pd.get_dummies(y_train),validation_split=0.1,epochs=10, batch_size=1)
```

    Epoch 1/10
    262/262 [==============================] - 1s 4ms/step - loss: 0.0132 - accuracy: 0.9962 - val_loss: 1.8289 - val_accuracy: 0.7667
    Epoch 2/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0107 - accuracy: 0.9962 - val_loss: 1.9073 - val_accuracy: 0.7667
    Epoch 3/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0107 - accuracy: 0.9962 - val_loss: 1.9267 - val_accuracy: 0.7667
    Epoch 4/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0084 - accuracy: 0.9962 - val_loss: 1.9633 - val_accuracy: 0.7667
    Epoch 5/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0089 - accuracy: 0.9962 - val_loss: 1.9930 - val_accuracy: 0.7667
    Epoch 6/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0083 - accuracy: 0.9962 - val_loss: 2.0202 - val_accuracy: 0.7667
    Epoch 7/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0084 - accuracy: 0.9962 - val_loss: 2.0428 - val_accuracy: 0.7667
    Epoch 8/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0086 - accuracy: 0.9962 - val_loss: 2.0780 - val_accuracy: 0.7667
    Epoch 9/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0089 - accuracy: 0.9962 - val_loss: 2.0998 - val_accuracy: 0.7667
    Epoch 10/10
    262/262 [==============================] - 1s 3ms/step - loss: 0.0110 - accuracy: 0.9962 - val_loss: 2.1445 - val_accuracy: 0.7667
    




    <keras.callbacks.History at 0x7f6f278caa10>




```python
model.evaluate(Xtest,pd.get_dummies(y_test))
```

    3/3 [==============================] - 0s 5ms/step - loss: 2.2551 - accuracy: 0.7027
    




    [2.2550535202026367, 0.7027027010917664]



Well this is not the state of art approach but still it is quite astonishing to see these cute results. Nonetheless we should try out word embeddings as well but it feels like this will be overkill. As a final remark before getting to Word Embeddings i would like to state that this area is actually very promising. There are millions of help center calls each and everyday and a system which can auto-direct user questions beforehand could be really useful.

# Naive CNN model


```python
# Padding the data samples to a maximum review length in words
max_words = 314
X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=max_words)
X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=max_words)
# Building the CNN Model
model = Sequential()      # initilaizing the Sequential nature for CNN model
# Adding the embedding layer which will take in maximum of 450 words as input and provide a 32 dimensional output of those words which belong in the top_words dictionary
model.add(Embedding(7000, 32, input_length=max_words))
model.add(Conv1D(64, 3, padding='same', activation='swish'))
model.add(MaxPooling1D())
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(256, activation='swish'))
model.add(Dense(128, activation='swish'))
model.add(Dense(15, activation='softmax'))
```


```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 314, 32)           224000    
    _________________________________________________________________
    conv1d (Conv1D)              (None, 314, 64)           6208      
    _________________________________________________________________
    max_pooling1d (MaxPooling1D) (None, 157, 64)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 157, 32)           6176      
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 78, 32)            0         
    _________________________________________________________________
    flatten (Flatten)            (None, 2496)              0         
    _________________________________________________________________
    dense_36 (Dense)             (None, 256)               639232    
    _________________________________________________________________
    dense_37 (Dense)             (None, 128)               32896     
    _________________________________________________________________
    dense_38 (Dense)             (None, 15)                1935      
    =================================================================
    Total params: 910,447
    Trainable params: 910,447
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.fit(X_train, pd.get_dummies(y_train), validation_split=0.1, epochs=10, batch_size=1, verbose=2)
```

    Epoch 1/10
    262/262 - 3s - loss: 0.6927 - accuracy: 0.7519 - val_loss: 2.0906 - val_accuracy: 0.5333
    Epoch 2/10
    262/262 - 2s - loss: 0.3531 - accuracy: 0.8588 - val_loss: 2.4082 - val_accuracy: 0.5000
    Epoch 3/10
    262/262 - 2s - loss: 0.2478 - accuracy: 0.9122 - val_loss: 2.9456 - val_accuracy: 0.5333
    Epoch 4/10
    262/262 - 2s - loss: 0.2561 - accuracy: 0.9122 - val_loss: 1.8203 - val_accuracy: 0.5667
    Epoch 5/10
    262/262 - 2s - loss: 0.3420 - accuracy: 0.9046 - val_loss: 2.4658 - val_accuracy: 0.5667
    Epoch 6/10
    262/262 - 2s - loss: 0.2860 - accuracy: 0.9008 - val_loss: 1.8315 - val_accuracy: 0.6333
    Epoch 7/10
    262/262 - 2s - loss: 0.2426 - accuracy: 0.9618 - val_loss: 1.9857 - val_accuracy: 0.6667
    Epoch 8/10
    262/262 - 2s - loss: 0.0536 - accuracy: 0.9847 - val_loss: 2.2933 - val_accuracy: 0.6667
    Epoch 9/10
    262/262 - 2s - loss: 0.0251 - accuracy: 0.9924 - val_loss: 2.4355 - val_accuracy: 0.6667
    Epoch 10/10
    262/262 - 2s - loss: 0.0204 - accuracy: 0.9962 - val_loss: 2.1803 - val_accuracy: 0.7000
    




    <keras.callbacks.History at 0x7f6f2792c2d0>




```python
model.evaluate(X_test,pd.get_dummies(y_test))
```

    3/3 [==============================] - 0s 10ms/step - loss: 2.3543 - accuracy: 0.5946
    




    [2.354349136352539, 0.5945945978164673]



Not that sexy right but it does not change the fact that this is bit fun to work.

# Non-Static CNN model


```python
def non_static_model():

    # channel 1

    inputs1 = Input(shape=(314, ))
    embedding1 = Embedding(input_dim=pretrained_weight_2.shape[0],
                           output_dim=400,
                           weights=[pretrained_weight_2],
                           trainable=True, input_length=450)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=2, activation='relu'
                   )(embedding1)
    pool1 = GlobalMaxPooling1D()(conv1)
    flat1 = Flatten()(pool1)

    
    conv2 = Conv1D(filters=32, kernel_size=3, activation='relu'
                   )(embedding1)
    pool2 = GlobalMaxPooling1D()(conv2)
    flat2 = Flatten()(pool2)

    
    conv3 = Conv1D(filters=32, kernel_size=4, activation='relu'
                   )(embedding1)
    pool3 = GlobalMaxPooling1D()(conv3)
    flat3 = Flatten()(pool3)

    # merge

    merged = concatenate([flat1,flat2,flat3])

    # interpretation
    dense = Dense(32, activation='relu')(merged)
    drop1 = Dropout(0.2)(dense)
    outputs = Dense(15, activation='softmax')(drop1)
    model = Model(inputs=[inputs1], outputs=outputs)

    # compile

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # summarize

    print(model.summary())
    return model
model = non_static_model()
model.fit(X_train, pd.get_dummies(y_train), validation_split=0.1, epochs=5, batch_size=1, verbose=2)
```

    Model: "model_6"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_14 (InputLayer)           [(None, 314)]        0                                            
    __________________________________________________________________________________________________
    embedding_8 (Embedding)         (None, 314, 400)     164982800   input_14[0][0]                   
    __________________________________________________________________________________________________
    conv1d_20 (Conv1D)              (None, 313, 32)      25632       embedding_8[0][0]                
    __________________________________________________________________________________________________
    conv1d_21 (Conv1D)              (None, 312, 32)      38432       embedding_8[0][0]                
    __________________________________________________________________________________________________
    conv1d_22 (Conv1D)              (None, 311, 32)      51232       embedding_8[0][0]                
    __________________________________________________________________________________________________
    global_max_pooling1d_12 (Global (None, 32)           0           conv1d_20[0][0]                  
    __________________________________________________________________________________________________
    global_max_pooling1d_13 (Global (None, 32)           0           conv1d_21[0][0]                  
    __________________________________________________________________________________________________
    global_max_pooling1d_14 (Global (None, 32)           0           conv1d_22[0][0]                  
    __________________________________________________________________________________________________
    flatten_19 (Flatten)            (None, 32)           0           global_max_pooling1d_12[0][0]    
    __________________________________________________________________________________________________
    flatten_20 (Flatten)            (None, 32)           0           global_max_pooling1d_13[0][0]    
    __________________________________________________________________________________________________
    flatten_21 (Flatten)            (None, 32)           0           global_max_pooling1d_14[0][0]    
    __________________________________________________________________________________________________
    concatenate_6 (Concatenate)     (None, 96)           0           flatten_19[0][0]                 
                                                                     flatten_20[0][0]                 
                                                                     flatten_21[0][0]                 
    __________________________________________________________________________________________________
    dense_51 (Dense)                (None, 32)           3104        concatenate_6[0][0]              
    __________________________________________________________________________________________________
    dropout_6 (Dropout)             (None, 32)           0           dense_51[0][0]                   
    __________________________________________________________________________________________________
    dense_52 (Dense)                (None, 15)           495         dropout_6[0][0]                  
    ==================================================================================================
    Total params: 165,101,695
    Trainable params: 165,101,695
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    Epoch 1/5
    262/262 - 427s - loss: 2.3720 - accuracy: 0.3206 - val_loss: 1.9882 - val_accuracy: 0.4333
    Epoch 2/5
    262/262 - 421s - loss: 1.4727 - accuracy: 0.5229 - val_loss: 1.8552 - val_accuracy: 0.6333
    Epoch 3/5
    262/262 - 421s - loss: 1.0607 - accuracy: 0.6641 - val_loss: 1.5124 - val_accuracy: 0.6000
    Epoch 4/5
    262/262 - 423s - loss: 0.7562 - accuracy: 0.7595 - val_loss: 1.7470 - val_accuracy: 0.5333
    Epoch 5/5
    262/262 - 421s - loss: 0.7227 - accuracy: 0.7634 - val_loss: 1.6775 - val_accuracy: 0.6667
    




    <keras.callbacks.History at 0x7f6f2535c690>




```python
model.evaluate(X_test,pd.get_dummies(y_test))
```

    3/3 [==============================] - 1s 47ms/step - loss: 1.3869 - accuracy: 0.6486
    




    [1.3868952989578247, 0.6486486196517944]



Results are not that cool hence we should select the simpler model. One promising fact is that we can extend this dataset by accessing other banks FAQ's and combine them with respect to their respective classes in the current dataset. With an extended dataset it is quite possible that we can achieve a much higher result set.


```python
def onehot2indices(encoded):
  decoded = []
  for line in encoded.iterrows():
    for index,l in enumerate(line[1]):
      if(l==1):
        decoded.append(index)
  return decoded
decoded = onehot2indices(y_test)
```


```python
conf = confusion_matrix(decoded,y_pred)
```


```python
plt.figure(figsize = (10,7))
sn.heatmap(conf, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f6f1e72a650>




    
![png](FAQ_Classification_files/FAQ_Classification_68_1.png)
    


The dataset i really small so it is not very clear whether the model works well since we are not able to check certain labels. I have to state that this is nothing other than a starting point. In order to obtain a valid work, the dataset should be extended. As a future work i will first try to expand the dataset and then use different CNN models discussed by Yoon Kim.
