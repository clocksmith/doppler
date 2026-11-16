"""Client-owned full-route cursor. Admission precedes encryption/disclosure; failure is terminal."""
from types import MappingProxyType


class FullRouteAdmission:
    def __init__(self,plan):
        value=plan.definition()
        self.schedule=tuple(MappingProxyType(dict(tile)) for tile in plan.schedule())
        self.steps,self.prefill_rows=value['steps'],value['contextBucket']
        self.position,self.in_flight,self.failed=0,False,False
        self.projection=None
        if not self.schedule or type(self.steps) is not int or self.steps<=0:
            raise ValueError('Complete encrypted route required')

    def fail(self):
        self.failed=True

    def _require(self,condition,message):
        if self.failed or not condition:
            self.failed=True
            raise ValueError('Encrypted route admission: '+message)

    def next_tile(self):
        self._require(self.position<len(self.schedule)*self.steps,'route exhausted or unavailable')
        return self.schedule[self.position%len(self.schedule)]

    def admit_projection(self,name,input_width,output_width,rows):
        expected=self.next_tile()
        expected_rows=self.prefill_rows if self.position<len(self.schedule) else 1
        self._require(not self.in_flight and self.projection is None and expected['name']==name
                      and expected['inputBegin']==0 and expected['outputBegin']==0
                      and type(rows) is int and rows==expected_rows,
                      'projection, retry or row schedule changed')
        projection=[tile for tile in self.schedule if tile['name']==name]
        self._require(input_width==max(tile['inputEnd'] for tile in projection)
                      and output_width==max(tile['outputEnd'] for tile in projection),
                      'projection geometry changed')
        self.projection=name

    def admit_tile(self,tile,matrix_shape,sequence):
        expected=self.next_tile()
        self._require(not self.in_flight and self.projection==expected['name'] and type(sequence) is int and sequence==self.position+1
                      and dict(tile)==dict(expected)
                      and tuple(matrix_shape)==(expected['inputEnd']-expected['inputBegin'],expected['outputEnd']-expected['outputBegin']),
                      'tile, attempt sequence or matrix geometry changed')
        self.in_flight=True
        return expected

    def verified(self):
        self._require(self.in_flight,'no admitted encrypted response')
        self.position+=1
        self.in_flight=False
        if self.position==len(self.schedule)*self.steps or self.schedule[self.position%len(self.schedule)]['name']!=self.projection or self.position%len(self.schedule)==0:
            self.projection=None

    def complete(self):
        self._require(not self.in_flight and self.position==len(self.schedule)*self.steps,'full encrypted route incomplete')
