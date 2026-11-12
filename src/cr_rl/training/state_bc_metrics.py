"""Count-weighted diagnostics, independent of the BC loss/gradient computation."""
class BCMetricWindow:
    ratios = {'exact_accuracy':'rows', 'wait_accuracy':'wait_rows', 'act_recall':'act_rows',
              'slot_accuracy':'play_rows', 'tile_mean_distance':'play_rows',
              'ability_accuracy':'ability_rows'}
    losses = ('loss','timing_loss','conditional_slot_loss','position_loss','card_loss')

    def __init__(self, every_updates=32, initial_step=0):
        if every_updates < 1:
            raise ValueError('Metric interval must be positive')
        self.every_updates=every_updates
        self.start_step=initial_step
        self._clear()

    def _clear(self):
        self.seats=0; self.groups=0; self.successes=0; self.seconds=0.; self.norm_sum=0.
        self.overflow=False; self.context={}; self.first_index=None
        self.single_seat={}
        self.counts=dict.fromkeys(('rows','act_rows','play_rows','wait_rows','ability_rows','history_frames'),0)
        self.numerators=dict.fromkeys(self.ratios,0.)
        self.loss_sums=dict.fromkeys(self.losses,0.)

    def add(self, report, **context):
        seats=report.get('reports',[report])
        self.context=context
        if self.first_index is None:
            self.first_index=context['episode_index']-len(seats)+1
        self.groups+=1
        self.successes+=int(report['optimizer_stepped'])
        self.seconds+=report['seconds']
        self.norm_sum+=report['grad_norm']
        self.overflow|=bool(report['amp_overflow_skipped'])
        for row in seats:
            self.seats+=1
            self.single_seat={key:row[key] for key in ('seat','replay_id') if key in row}
            counts={key:row.get(key,0) for key in self.counts}
            counts['ability_rows']=max(0,row['act_rows']-row['play_rows'])
            for key,value in counts.items():
                self.counts[key]+=value
            for key,denominator in self.ratios.items():
                if counts[denominator]:
                    self.numerators[key]+=row[key]*counts[denominator]
            for key in self.losses:
                self.loss_sums[key]+=row[key]

    def ready(self):
        return bool(self.seats and (self.context['optimizer_steps']-self.start_step>=self.every_updates or self.overflow))

    def flush(self):
        if not self.seats:
            return None
        result={'phase':'state_bc_train', **self.context, **self.counts,
                'aggregation':'optimizer_update_window_v1', 'window_seats':self.seats,
                'window_groups':self.groups, 'window_optimizer_updates':self.successes,
                'episode_index_start':self.first_index, 'window_seconds':self.seconds,
                'seconds':self.seconds/self.seats, 'group_seconds':self.seconds/self.groups,
                'grad_norm':self.norm_sum/self.groups, 'amp_overflow_skipped':self.overflow,
                'optimizer_stepped':self.successes>0, 'batch_optimizer_stepped':self.successes>0}
        result.update({key:value/self.seats for key,value in self.loss_sums.items()})
        if self.seats==1:
            result.update(self.single_seat)
        for key,denominator in self.ratios.items():
            if self.counts[denominator]:
                result[key]=self.numerators[key]/self.counts[denominator]
        self.start_step=self.context['optimizer_steps']
        self._clear()
        return result
