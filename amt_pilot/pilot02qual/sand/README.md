
**Qualifications**
How to add the data protection thingy as qualification to our HITs:
 * edit text in question form overview (now says "bla bla"), make sure that it is valid xml
 * run:
    > python ../../amt_api/createqual.py config.ini
 * this will create a json with the ID for the Qualification Type (QualificationTypeId)
 * ... add this ID to the config file in section [qualification], e.g.
 
 [qualification]
  approvedhits = 0
  approvalrate = 0
  protectionid = 39XUM4N2MFI9OD5015XCF9E8Z9ZHK8 (<- change this ID)
