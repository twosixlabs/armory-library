import { computed } from 'vue';
import { useRouter } from 'vue-router';
import Button from './button.js';
import { XMarkIcon } from './icons.js';
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
        Button,
        Table,
        TableBody,
        TableCell,
        TableHead,
        TableHeader,
        TableRow,
        TableRowHeader,
        XMarkIcon,
    },
    props: {
        run: Object,
    },
    setup(props) {
        const router = useRouter();
        const route = router.currentRoute;

        const filter = computed({
            get() {
                const filter = route.value.query.filter;
                if (filter == undefined) {
                    return "";
                }
                return filter;
            },
            set(filter) {
                router.push(
                    { query: { ...route.value.query, filter }},
                    { replace: true },
                );
            },
        });

        const params = computed(() => {
            const params = Object.entries(props.run.data.params);
            if (filter.value.length > 0) {
                return params.filter(([name, _]) => name.includes(filter.value));
            }
            return params;
        });


        return { filter, params };
    },
    template: `
        <div class="items-center flex flex-row gap-2 my-2">
            <span>
                Filter
            </span>
            <input
                v-model="filter"
                class="appearance-none border border-zinc-300 focus:border-zinc-400 focus:outline-none leading-6 pl-3 pr-2 py-1.5 rounded-md w-80"
            />
            <Button @click="filter = undefined" minimal>
                <XMarkIcon></XMarkIcon>
            </Button>
        </div>
        <Table>
            <TableHead>
                <tr>
                    <TableHeader>Parameter</TableHeader>
                    <TableHeader>Value</TableHeader>
                </tr>
            </TableHead>
            <TableBody>
                <TableRow v-for="[name, value] in params" :key="name">
                    <TableRowHeader>
                        {{ name }}
                    </TableRowHeader>
                    <TableCell>
                        {{ value }}
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
