import { defineStore } from 'pinia';

export const useSelectedRuns = defineStore('selected-runs', {
    state: () => ({
        runs: [],
    }),
    actions: {
        selectRuns(runs) {
            this.runs = [...runs];
        },
        singleSelect(run) {
            this.runs = [run];
        },
        multiSelect(run) {
            const index = this.runs.indexOf(run);
            if (index == -1) {
                this.runs.push(run);
            } else {
                this.runs.splice(index, 1);
            }
        },
    }
});
