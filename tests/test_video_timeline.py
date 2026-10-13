import json
from unittest.mock import patch
import pytest
from vibetrack.viewers.web import _video_timeline

def test_variable_rate_timestamps_and_cache_invalidation():
    _video_timeline.cache_clear()
    payload={'frames':[{'best_effort_timestamp_time':t} for t in ['2.0','2.04','2.12','2.14']]}
    with patch('vibetrack.viewers.web.subprocess.run') as run:
        run.return_value.stdout=json.dumps(payload)
        a=_video_timeline('/video.mp4',1,100)
        assert a['times']==pytest.approx([0,.04,.12,.14])
        assert a['frames']==4
        _video_timeline('/video.mp4',1,100)
        assert run.call_count==1
        _video_timeline('/video.mp4',2,100)
        assert run.call_count==2

@pytest.mark.parametrize('times',[[],['nan'],['0','0'],['1','0']])
def test_reject_invalid_timeline(times):
    _video_timeline.cache_clear()
    with patch('vibetrack.viewers.web.subprocess.run') as run:
        run.return_value.stdout=json.dumps({'frames':[{'best_effort_timestamp_time':t} for t in times]})
        with pytest.raises(ValueError):
            _video_timeline('/video.mp4',1,100)
