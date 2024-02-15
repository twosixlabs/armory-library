import { computed } from 'vue';
import {
    formatDuration,
    formatTime,
    humanizeDuration,
    humanizeTime,
} from '../utils/format.js';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
    TableRowHeader,
} from './table.js';

export default {
    components: {
        Table,
        TableBody,
        TableCell,
        TableHead,
        TableHeader,
        TableRow,
        TableRowHeader,
    },
    props: {
        run: Object,
    },
    setup(props) {
        const rows = computed(() => {
            const rows = [];
            rows.push(["Name", props.run.info.run_name]);
            rows.push(["Description", props.run.data.tags["mlflow.note.content"]]);

            if (props.run.info.start_time) {
                const exact = formatTime(props.run.info.start_time);
                const human = humanizeTime(props.run.info.start_time);
                rows.push(["Started", `${exact} (${human})`]);
            } else {
                rows.push(["Started", ""]);
            }

            if (props.run.info.end_time) {
                const exact = formatTime(props.run.info.end_time);
                const human = humanizeTime(props.run.info.end_time);
                rows.push(["Ended", `${exact} (${human})`]);
            } else {
                rows.push(["Ended", ""]);
            }

            if (props.run.info.start_time && props.run.info.end_time) {
                const exact = formatDuration(props.run.info.end_time - props.run.info.start_time);
                const human = humanizeDuration(props.run.info.end_time - props.run.info.start_time);
                rows.push(["Duration", `${exact} (${human})`]);
            } else {
                rows.push(["Duration", ""]);
            }

            return rows;
        });
        return { rows };
    },
    template: `
        <Table class="mt-2">
            <TableHead>
                <tr>
                    <TableHeader>Key</TableHeader>
                    <TableHeader>Value</TableHeader>
                </tr>
            </TableHead>
            <TableBody>
                <TableRow>
                    <TableRowHeader colspan="2">
                        Summary
                    </TableRowHeader>
                </TableRow>
                <TableRow
                    v-for="[key, value] in rows"
                    :key="key"
                >
                    <TableRowHeader>
                        <span class="ml-4">
                            {{ key }}
                        </span>
                    </TableRowHeader>
                    <TableCell>
                        {{ value }}
                    </TableCell>
                </TableRow>

                <TableRow>
                    <TableRowHeader colspan="2">
                        Info
                    </TableRowHeader>
                </TableRow>
                <TableRow
                    v-for="[key, value] in Object.entries(run.info)"
                    :key="key"
                >
                    <TableRowHeader>
                        <span class="ml-4">
                            {{ key }}
                        </span>
                    </TableRowHeader>
                    <TableCell>
                        {{ value }}
                    </TableCell>
                </TableRow>

                <TableRow>
                    <TableRowHeader colspan="2">
                        Tags
                    </TableRowHeader>
                </TableRow>
                <TableRow
                    v-for="[key, value] in Object.entries(run.data.tags)"
                    :key="key"
                >
                    <TableRowHeader>
                        <span class="ml-4">
                            {{ key }}
                        </span>
                    </TableRowHeader>
                    <TableCell>
                        {{ value }}
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
