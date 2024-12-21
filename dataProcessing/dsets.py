from collections import namedtuple

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple', 
    ['isNodule_bool', 'diameter_mm', 'noduleID', 'center_xyz']
    )
