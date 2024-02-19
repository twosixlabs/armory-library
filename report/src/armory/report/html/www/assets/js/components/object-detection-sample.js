import { computed } from 'vue';
import { argmax } from '../utils/argmax.js';

export default {
    props: {
        batch: Number,
        lhsChain: String,
        rhsChain: String,
        run: Object,
        sample: Number,
    },
    setup(props) {
        const lhsImage = computed(() => {
            const file = props.run.artifacts[props.lhsChain]?.[props.batch]?.[props.sample]?.file;
            if (file) {
                return `./assets/img/${props.run.info.run_id}/${file}`;
            }
            return "";
        });

        const rhsImage = computed(() => {
            const file = props.run.artifacts[props.rhsChain]?.[props.batch]?.[props.sample]?.file;
            if (file) {
                return `./assets/img/${props.run.info.run_id}/${file}`;
            }
            return "";
        });

        return {
            lhsImage,
            rhsImage,
        };
    },
    template: `
        <div class="flex gap-4 items-center justify-center">
            <div class="flex flex-col">
                <span>{{ lhsChain }}</span>
            </div>
            <div v-if="lhsImage && rhsImage" class="diff aspect-square max-w-xl">
                <div class="diff-item-1">
                    <img :src="rhsImage" />
                </div>
                <div class="diff-item-2">
                    <img :src="lhsImage" />
                </div>
                <div class="diff-resizer"></div>
            </div>
            <div class="flex flex-col">
                <span>{{ rhsChain }}</span>
            </div>
        </div>
    `,
};
