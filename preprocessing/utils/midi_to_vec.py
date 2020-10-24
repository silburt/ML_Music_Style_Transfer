# taken from https://medium.com/analytics-vidhya/convert-midi-file-to-numpy-array-in-python-7d00531890c
# TODO: working well, but I think need to handle all the tempo changes in these files, otherwise the alignment goes off. 

import mido
import string
import numpy as np


def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]


def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result


def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]
    
    
def track2seq(track, bins_per_second, tempos):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        msg = track[i]
        new_state, new_time = get_new_state(msg, last_state)
        if new_time > 0:
            result += [last_state] * new_time
        last_state, last_time = new_state, new_time
    return result

# ---------------------------------------------------

def get_state(note):
    note_value = note['value']
    state = [0] * 88
    if 21 <= note_value <= 108:
        state[note_value - 21] = note['velocity'] if note['on'] is True else 0
    return state


def notes_to_pianoroll(notes, tempos, ticks_per_beat, bins_per_second):
    current_tempo = tempos.pop()
    next_tempo = tempos.pop() if len(tempos) > 0 else None

    previous_time_seconds = 0
    previous_state = [0] * 88

    pianoroll = []
    for note in notes:
        # configure tempos
        if next_tempo is not None and note['absolute_time'] >= next_tempo['absolute_time']:
            current_tempo = next_tempo
            next_tempo = tempos.pop() if len(tempos) > 0 else None
        
        # get delta_time of message
        next_time_seconds = mido.tick2second(note['delta_time'], ticks_per_beat, current_tempo['tempo'])
        
        # build piano roll up to current time using previous states
        n_previous_state_bins = int((next_time_seconds - previous_time_seconds) * bins_per_second)
        for i in range(n_previous_state_bins):
            pianoroll.append(previous_state)

        # add new state
        previous_state = get_state(note)    # this is the "current state", but automatically overwriting previous_state
        pianoroll.append(previous_state)
    return np.asarray(pianoroll)


def extract_all_notes(mid):
    notes = []
    for track in mid.tracks:
        absolute_time = 0
        for msg in track:
            absolute_time += int(msg.time)
            try:
                if 'note' in msg.type:
                    on_ = False if 'note_off' in msg.type else True
                    notes.append(
                        {
                            'absolute_time': absolute_time, 
                            'delta_time':int(msg.time), 
                            'velocity': msg.velocity, 
                            'value': msg.note, 
                            'on': on_
                        })
            except:
                print('couldnt process msg:', msg)

    # sort tempos smallest to largest
    notes = sorted(notes, key=lambda x: x['absolute_time'])
    return notes


def extract_all_tempos(mid):
    tempos = []
    for track in mid.tracks:
        absolute_time = 0
        for msg in track:
            try:
                if 'set_tempo' in msg.type:
                    absolute_time += int(msg.time)
                    tempos.append({'absolute_time': absolute_time, 'tempo': msg.tempo})
            except:
                print('couldnt process msg:', msg)

    if len(tempos) == 0:
        raise ValueError('could not find any tempos from midi file!!')

    # sort tempos in absolute, earliest is at the end, for easy .pop() later
    tempos = sorted(tempos, key=lambda x: x['absolute_time'], reverse=True) 
    return tempos


def get_midi_ticks_per_beat(mid):
    return mid.ticks_per_beat


def midi_file_to_array(midi_file, min_msg_pct=0.1, bins_per_second=250):
    '''
    bins_per_second = 1/0.004 -> each bin is 4ms
    '''
    mid = mido.MidiFile(midi_file)
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    
    # get ticks per beat
    ticks_per_beat = get_midi_ticks_per_beat(mid)

    # extract all tempos
    tempos = extract_all_tempos(mid)

    # extract all notes
    notes = extract_all_notes(mid)

    # get pianoroll
    piano_roll = notes_to_pianoroll(notes, tempos, ticks_per_beat, bins_per_second)

    '''
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i], bins_per_second, tempos)
            all_arys.append(ary_i)
    
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]'''
    return piano_roll, ticks_per_beat, bins_per_second


def pianoroll_to_midi_file(piano_roll, mid_filepath, ticks_per_beat, bins_per_second, tempo_bpm=120):
    tempo = mido.bpm2tempo(tempo_bpm)

    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    mid_new.ticks_per_beat = ticks_per_beat

    previous_state = np.asarray([0] * 88)
    current_time = 0
    last_message_time = 0
    for i, state in enumerate(piano_roll):
        # update time
        current_time += int(mido.second2tick(1 / bins_per_second, ticks_per_beat, tempo))

        # send msg for new state
        diff = state - previous_state
        if sum(diff) > 0:   # message
            on_notes = np.where(diff > 0)[0]
            on_notes_vol = state[on_notes]
            off_notes = np.where(diff < 0)[0]
            
            # time
            delta_time = current_time - last_message_time

            #print(i, last_message_time on_notes, on_notes_vol, off_notes)

            # add messages
            for n, v in zip(on_notes, on_notes_vol):
                track.append(mido.Message('note_on', note=n + 21, velocity=v, time=delta_time))
            for n in off_notes:
                track.append(mido.Message('note_off', note=n + 21, velocity=0, time=delta_time))

            last_message_time = current_time

        # update state
        previous_state = state

    mid_new.save(mid_filepath)


def array_to_midi_file(ary, mid_filepath, tempo_bpm=120):
    # bpm tempo to midi tempo
    tempo = int( 2 * mido.bpm2tempo(tempo_bpm) )

    # get the difference
    new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
    changes = new_ary[1:] - new_ary[:-1]
    # create a midi file with an empty track
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    # add difference in the empty track
    last_time = 0
    for ch in changes:
        if set(ch) == {0}:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_vol = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_vol):
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
                first_ = False
            last_time = 0
    
    mid_new.save(mid_filepath)
