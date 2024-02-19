import { computed } from 'vue';
import { TableCell } from './table.js';

export const BETTER_THAN_BASELINE = 1;
export const SAME_AS_BASELINE = 0;
export const WORSE_THAN_BASELINE = -1;

export const MetricCell = {
    components: {
        TableCell,
    },
    props: {
        comparison: Number,
        precision: Number,
        value: Number,
    },
    setup(props) {
        const classes = computed(() => ({
            cell: {
                "text-green-700": props.comparison == BETTER_THAN_BASELINE,
                "text-red-700": props.comparison == WORSE_THAN_BASELINE,
            },
            span: {
                "border-r-[1rem]": props.comparison != SAME_AS_BASELINE,
                "border-green-700": props.comparison == BETTER_THAN_BASELINE,
                "border-red-700": props.comparison == WORSE_THAN_BASELINE,
                "pr-1": props.comparison != SAME_AS_BASELINE,
            },
        }));
        return { classes };
    },
    template: `
        <TableCell :class="classes.cell">
            <span :class="classes.span">
                {{ value?.toFixed(precision) }}
            </span>
        </TableCell>
    `,
};
