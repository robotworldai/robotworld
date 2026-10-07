from environment.runtime.continuation import remember_packet, continue_packet


class Policy:
    step = 0
    samples = 0

    @remember_packet(lambda self:self.step)
    def content(self):
        self.samples += 1
        return [{'type':'inputText','text':f'CURRENT step {self.step}'},
                {'type':'inputImage','imageUrl':'unchanged pixels'}]


def test_text_continuations_do_not_resample_or_mutate_packet():
    policy=Policy()
    original=policy.content()
    for _ in range(30):
        packet=continue_packet(policy,policy.step,policy.content)
        assert packet[1:]==original
        packet[1]['text']='caller mutation'
    assert policy.samples==1


def test_new_step_same_pixels_is_not_hash_deduplicated():
    policy=Policy();policy.content();policy.step=1
    packet=continue_packet(policy,policy.step,policy.content)
    assert policy.samples==2 and packet[1]['text']=='CURRENT step 1'


def test_all_seven_policy_families_use_shared_continuation():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'benchmarks'
    for bench in ('robocasa','robolab','behavior_1k','ai_cps','wheeledlab','humanoid_soccer','native_project'):
        text=(root/bench/'policy.py').read_text()
        assert '@remember_packet(' in text and 'continue_packet(self,' in text
