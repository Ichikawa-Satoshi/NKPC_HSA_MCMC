*****************************************************
TOP LEVEL NOTE: TNIC HHI and Total Similarity are both measures of market structure and market power.  HHI is positively associated with pricing power according to theory 
(for example consider a Cournot oligopoly model), and total similarity is negatively related according to product differentiation theory (consider Chamberlin or Hotelling).
We note that both HHIs and Total similarities in this database are customized to each firm.  This is the case because the TNIC industry classification is also firm-specific,
and every firm has its unique set of rivals.  See papers referenced below for more detail.

****** NOTE: Please read the technical descriptions below before using the data.  


******************************

This file accompanies the TNIC-3 industry concentration database and describes where the data comes from,
the papers that should be cited when providing academic references, and some very important technical details regarding its usage.
Please read the technical details in full before using this data.  These details are critically important to ensure proper usage.
The data is at the firm-year level.  

This file includes both total similarity scores and also HHIs for each firm.  See technical notes below, and also consider reading the
paper titled "Text-Based Network Industries and Endogenous Product Differentiation" referred to below for more details.  We provide a 
basic overview of these variables below in the technical descriptions area.


**********************************************************************************************************************************
**********************************************************************************************************************************
******************************************* General Background on TNIC industries *************************************************
******************************************* General Background on TNIC industries *************************************************
******************************************* General Background on TNIC industries *************************************************
**********************************************************************************************************************************
**********************************************************************************************************************************

For an extensive description of this data, please read the data and methodology sections of the studies noted below.  Here is a 
brief description.

This data is based on web crawling and text parsing algorithms that process the text in the business descriptions of 10-K annual 
filings on the SEC Edgar website from 1996 to present.  These product descriptions are legally required to be accurate, as Item 101 
of Regulation S-K legally requires that firms describe the significant products they offer to the market, and these descriptions 
must also be updated and representative of the current fiscal year of the 10-K.  We merge each firm's text product description to 
the CRSP/COMPUSTAT universe using the central index key (CIK) [We thank the Wharton Research Data Service (WRDS) for providing us 
with an expanded historical mapping of SEC CIK to COMPUSTAT gvkey, as the base CIK variable in COMPUSTAT only contains current links].  
Our resulting database is based on all publicly traded firms (domestic firms traded on either NYSE, AMEX, or NASDAQ) for which we have 
COMPUSTAT and CRSP data.

We calculate our firm-by-firm pairwise similarity scores by parsing the product descriptions from the firm 10Ks and forming word vectors 
for each firm to compute continuous measures of product similarity for every pair of firms in our sample in each year (a pairwise 
similarity matrix).  This is done using the cosine similarity method, which is applied after basic screens to eliminate common words are
applied (see studies noted below).   For any two firms i and j, we thus have a product similarity, which is a real number in the 
interval [0,1] describing how similar the words used by firms i and j are.

The TNIC-3 classification data we are distributing only records firms having pairwise similarities with a given firm i that are 
above a threshold as required based on the coraseness of the three digit SIC classification.  The level of coarseness of TNIC-3 thus matches 
that of three digit SIC codes, as both classifications result in the same number of firm pairs being deemed related.  For example, if one picks two
firms at random from the CRSP/COMPUSTAT universe, the likelihood of them being in the same three digit SIC code is 2.05%.  Analgously, when the TNIC-3
cutoff is specified using our approach, the likelihood of two randomly drawn firms being deemed related in their TNIC-3 is also 2.05%.  Hence, 
TNIC-3 is constructed to be "as coarse" as are three digit SIC codes.

Note:  TNIC industries are also purged for vertical relationships from the input/output tables (see paper for details).
Note 2: The words used to construct TNIC industries only include nouns or proper nouns (see paper for details) and we exclude geographic terms.

**************************************************************************************************************
**************************************************************************************************************
********************************************** Citations *****************************************************
********************************************** Citations *****************************************************
********************************************** Citations *****************************************************
**************************************************************************************************************
**************************************************************************************************************

Please cite the following study when using this HHI data:

Text-Based Network Industries and Endogenous Product Differentiation
Gerard Hoberg and Gordon Phillips, Journal of Political Economy(October 2016), 124 (5) 1423-1465.

* If using TNIC data beyond simple controls for Total similarities or HHIs, please consider citing the 2010 RFS study referred to in the readme file associated with TNIC industry classification data.

**********************************************************************************************************************
**********************************************************************************************************************
********************************************** Technical Details *****************************************************
********************************************** Technical Details *****************************************************
********************************************** Technical Details *****************************************************
**********************************************************************************************************************
**********************************************************************************************************************

Please read the following carefully to ensure proper usage of this data.

Technical Note 1) Each file contains a gvkey and year firm identifier.  The TNIC3HHI variable is a concentration measure and TNIC3TSIMM is a 
total similarity measure.  Each observation should be mapped to COMPUSTAT using fiscal year endings that match the year field in this data.  It is important to note that were already did the merge to COMPUSTAT, so you do not have to repeat this, which is why we provide data with gvkey as the identifier.  The data contained here is not lagged.  Researchers needing lagged data must lag the data on their own.  On date conventions, for convenience, the year field in this database is based on Compustat calendar years  
obtained as the first four digits of the YYYYMMDD datadate variable.  Consider a COMPUSTAT firm with a fiscal year ending on Sept 30th, 1997, for example.  The corresponding record for this firm's gvkey in 1997 in this database ia based on the product description of the 10-K report that was associated with this firm's 9/30/1997 fiscal year end.  

Technical Note 2) These HHI and total similarity data are computed using TNIC designations that include the firm itself in part of the HHI calculation.
All HHIs are based on firm sales data from COMPUSTAT, and are computed using the Herfindahl-Hirschmann sum of squared market shares formulation.

Technical Note 3) For a very small fraction of the database, the TNIC3HHI variable will be missing but the TNIC3TSIMM variable will be non-missing.  This occurs when we do not have 
Compustat sales data available for the focal firm.  We avoid computing an HHI when this is the case, but in rare cases, we still do have TNIC data for such firms and we leave these in the 
database because there is adequate data to compute a TNIC3TSIMM value in these cases (Total similarities do not rely on sales).

